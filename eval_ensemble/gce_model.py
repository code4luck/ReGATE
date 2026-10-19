import torch
import torch.nn as nn
import torch.nn.functional as F

from pooling_head import QueryAttentionPooling

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

        # enzyme pooling
        self.enzyme_pooling = QueryAttentionPooling(self.protein_embedding_size, dropout=self.dropout)
        # mol pooling
        self.mol_pooling = QueryAttentionPooling(self.mol_embedding_size, dropout=self.dropout)

        # enzyme encoder
        self.enzyme_encoder = nn.Sequential(
            nn.LayerNorm(self.protein_embedding_size),
            nn.Linear(self.protein_embedding_size, self.hidden_size),
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
        # 暂时先不使用ec
        e_feat = self.enzyme_pooling(protein_embedding, protein_mask) # [B, D]
        m_feat = self.mol_pooling(mol_embedding, mol_mask) # [B, D]

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

        #
        expert_preds = torch.stack([y_e, y_m, y_f], dim=-1)  # [B, 3]
        gate_in = torch.cat([e_feat, m_feat, expert_preds], dim=-1)
        alpha = torch.softmax(self.gate(gate_in), dim=-1)   # [B, 3]
        pred = (alpha * expert_preds).sum(dim=-1)

        return {
            "pred": pred,
            "y_e": y_e,
            "y_m": y_m,
            "y_f": y_f,
            "alpha": alpha,
        }
