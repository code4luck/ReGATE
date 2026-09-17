import torch
import torch.nn as nn
import torch.nn.functional as F

from pooling_head import QueryAttentionPooling
from ec_model import HierECResidualEmbedding

class LowRankCrossLayer(nn.Module):
    def __init__(self, dim, rank=128, dropout=0.0):
        super().__init__()
        self.v = nn.Linear(dim, rank, bias=False)
        self.u = nn.Linear(rank, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, xl):
        # [B, D] -> [B, rank] -> [B, D]
        cross = self.u(self.dropout(self.v(xl)))
        return xl + x0 * cross

class LowRankCrossNetwork(nn.Module):
    def __init__(self, dim, num_layers=2, rank=128, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            LowRankCrossLayer(dim, rank, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x0):
        xl = x0
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl

class GatedCrossExpertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.protein_embedding_size = config.protein_embedding_size
        self.mol_embedding_size = config.mol_embedding_size
        self.dropout = config.dropout
        self.cross_layers = config.cross_layers
        self.cross_rank = config.cross_rank
        self.macc_dim = getattr(config, "macc_dim", 167)
        self.ec_dim = getattr(config, "ec_dim", 128)
        freeze_ec = getattr(config, "freeze_ec", False)
        pretrained_ec_encoder_path = getattr(config, "pretrained_ec_encoder_path", None)
        print(f"freeze_ec: {freeze_ec}")

        # enzyme pooling
        self.enzyme_pooling = QueryAttentionPooling(self.protein_embedding_size, dropout=self.dropout)
        # mol pooling
        self.mol_pooling = QueryAttentionPooling(self.mol_embedding_size, dropout=self.dropout)

        # ec_hier
        self.ec_module = HierECResidualEmbedding(
            vocab_sizes=config.vocab_sizes,
            dec_dim=self.ec_dim,
            dropout=self.dropout,
        )
        self.ec_proj = nn.Sequential(
            nn.LayerNorm(self.ec_dim),
            nn.Linear(self.ec_dim, self.ec_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # enzyme encoder
        self.enzyme_encoder = nn.Sequential(
            nn.LayerNorm(self.protein_embedding_size+self.ec_dim),
            nn.Linear(self.protein_embedding_size+self.ec_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.mol_encoder = nn.Sequential(
            nn.LayerNorm(self.mol_embedding_size + self.macc_dim),
            nn.Linear(self.mol_embedding_size + self.macc_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.enzyme_head = nn.Linear(self.hidden_size, 1)
        self.mol_head = nn.Linear(self.hidden_size, 1)

        d_pair = self.hidden_size * 2
        self.cross_net = LowRankCrossNetwork(
            dim=d_pair,
            num_layers=self.cross_layers,
            rank=self.cross_rank,
            dropout=self.dropout,
        )

        # self.deep_net = nn.Sequential(
        #     nn.LayerNorm(d_pair),
        #     nn.Linear(d_pair, self.hidden_size),
        #     nn.GELU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.GELU(),
        # )
        # self.fusion_head = nn.Sequential(
        #     nn.LayerNorm(d_pair + self.hidden_size),
        #     nn.Linear(d_pair + self.hidden_size, self.hidden_size),
        #     nn.GELU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden_size, 1),
        # )

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2 + 3),
            nn.Linear(self.hidden_size * 2 + 3, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 3),
        )

        # load ec_module weights
        if pretrained_ec_encoder_path is not None:
            payload = torch.load(pretrained_ec_encoder_path, map_location="cpu")
        else:
            raise ValueError("pretrained_ec_encoder_path is required for GatedCrossExpertModel.")
        state_dict = payload.get("state_dict", payload)
        self.ec_module.load_state_dict(state_dict, strict=True)
        if freeze_ec:
            self._freeze_ec_module()

    def _freeze_ec_module(self):
        for param in self.ec_module.parameters():
            param.requires_grad = False

    def pool_multiple_ec(
        self,
        ec_features,
        ec_mask,
        ec_owner,
        batch_size,
    ):
        """
        ec_features: [B', D_ec]
        ec_mask:     [B', 4]
        ec_owner:    [B']
        return:      [B, D_ec]
        """

        # 至少有一个已知层级的EC才参与均值
        valid_ec = ec_mask.any(dim=1).to(ec_features.dtype)  # [B']

        weighted_features = ec_features * valid_ec.unsqueeze(-1)
        pooled = ec_features.new_zeros(batch_size, ec_features.size(-1))
        pooled.index_add_(0, ec_owner, weighted_features)
        
        counts = ec_features.new_zeros(batch_size)
        counts.index_add_(0, ec_owner, valid_ec)
        
        pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)
        return pooled


    def _build_ec_features(
        self,
        batch_size,
        ec_ids,
        ec_mask,
        ec_owner,
    ):

        ec_feat = self.ec_module(ec_ids, ec_mask)  # [B', Dec]

        batch_size = batch_size
        ec_feat = self.pool_multiple_ec(
            ec_features=ec_feat,
            ec_mask=ec_mask,
            ec_owner=ec_owner,
            batch_size=batch_size,
        )  # [B, D_ec]
        ec_feat = self.ec_proj(ec_feat)  # [B, D]

        return ec_feat


    def forward(self,         
            protein_embedding, # [B, L, D]
            protein_mask, # 1 is true , 0 is padding
            mol_embedding, # [B, L, D]
            mol_mask,
            ec_ids=None, # [B', 4]
            ec_mask=None,
            ec_owner=None,
            macckeys=None, # [B, D]
        ):
    
        e_feat = self.enzyme_pooling(protein_embedding, protein_mask) # [B, D]
        m_feat = self.mol_pooling(mol_embedding, mol_mask) # [B, D]

        ec_feat = self._build_ec_features(
            batch_size=protein_embedding.size(0),
            ec_ids=ec_ids,
            ec_mask=ec_mask,
            ec_owner=ec_owner,
        )
        e_feat = torch.cat([e_feat, ec_feat], dim=-1) # [B, Dp + Dec]

        e_feat = self.enzyme_encoder(e_feat) # [B, D]
        # macc+m_feat
        m_feat = torch.cat([m_feat, macckeys], dim=-1)
        m_feat = self.mol_encoder(m_feat) # [B, D]
        y_e = self.enzyme_head(e_feat).squeeze(-1) # [B,]
        y_m = self.mol_head(m_feat).squeeze(-1) # [B,]

        pair = torch.cat([e_feat, m_feat], dim=-1)
        x_cross = self.cross_net(pair) # [B, D]
        
        # x_deep = self.deep_net(pair) # [B, D]
        # fusion_in = torch.cat([x_cross, x_deep], dim=-1)
        # y_f = self.fusion_head(fusion_in).squeeze(-1) # [B,]
        
        y_f = self.fusion_head(x_cross).squeeze(-1) # [B,]

        # gate 可以看到 e, m, 和三个 expert prediction
        expert_preds = torch.stack([y_e, y_m, y_f], dim=-1)  # [B, 3]
        gate_in = torch.cat([e_feat, m_feat, expert_preds], dim=-1)
        # gate_in = torch.cat([e_feat, m_feat], dim=-1)
        alpha = torch.softmax(self.gate(gate_in), dim=-1)   # [B, 3]
        pred = (alpha * expert_preds).sum(dim=-1)

        return {
            "pred": pred,
            "y_e": y_e,
            "y_m": y_m,
            "y_f": y_f,
            "alpha": alpha,
        }