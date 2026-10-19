
import torch
import torch.nn as nn

PAD, UNK, DASH = 0, 1, 2

class HierECResidualEmbedding(nn.Module):
    """
    z = e1 + sum_{i=2..4} mask_i * proj_i(e_i)
    """
    def __init__(self, vocab_sizes, dec_dim=64, dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(vocab_sizes[i], dec_dim, padding_idx=PAD) for i in range(4)
        ])
        self.projs = nn.ModuleList([nn.Identity()] + [
            nn.Sequential(nn.Linear(dec_dim, dec_dim), nn.GELU()) for _ in range(3)
        ])
        # self.level_scale = nn.Parameter(torch.tensor([1.0, 0.7, 0.5, 0.3]))
        self.out_norm = nn.LayerNorm(dec_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = dec_dim

    def forward(self, ec_ids, ec_mask):
        # ec_ids:  [B', 4]
        # ec_mask: [B', 4]
        z = torch.zeros(
            ec_ids.size(0),
            self.out_dim,
            device=ec_ids.device,
            dtype=self.embs[0].weight.dtype,
        )
        for level in range(4):
            embedding = self.embs[level](ec_ids[:, level])
            embedding = self.projs[level](embedding)

            gate = ec_mask[:, level].unsqueeze(-1)
            z = z + gate * embedding
        has_known_ec = ec_mask.any(dim=1, keepdim=True)
        z = self.dropout(self.out_norm(z))

        z = z * has_known_ec.to(z.dtype)
        return z  # [B', D_ec]

