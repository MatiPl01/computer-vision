import torch
import torch.nn as nn

# Configuration constants
DEBUG = False
DROPOUT = 0.1
NUM_HEADS = 8
FFN_MULT = 4
T = 300  # Number of time frames
NUM_JOINTS = 17  # Number of skeleton joints (e.g., COCO format)
JOINT_DIM = 3  # Joint dimensions (x, y, z coordinates)
H = 256  # Hidden dimension
NUM_CLASSES = 5  # Number of gymnastic exercise classes


class SpatialAttentionBlock(nn.Module):

    def __init__(self, H, num_heads=8, dropout=DROPOUT):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=H, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(H)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, J, H)
        B, T, J, H = x.shape
        if DEBUG:
            print(f"SpatialAtt in: {x.shape}")
        xt = x.reshape(B * T, J, H)  # (B*T, J, H) treating each frame independently
        # MHA expects (batch, seq, embed) with batch_first=True
        out, _ = self.mha(xt, xt, xt, need_weights=False)
        out = self.dropout(out)
        out = out + xt  # residual
        out = self.norm(out)
        out = out.reshape(B, T, J, H)
        if DEBUG:
            print(f"SpatialAtt out: {out.shape}")
        return out


class TemporalAttentionBlock(nn.Module):

    def __init__(self, H, num_heads=8, dropout=DROPOUT):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=H, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(H)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, J, H)
        B, T, J, H = x.shape
        if DEBUG:
            print(f"TemporalAtt in: {x.shape}")
        xt = x.permute(0, 2, 1, 3).reshape(B * J, T, H)  # (B*J, T, H)
        out, _ = self.mha(xt, xt, xt, need_weights=False)
        out = self.dropout(out)
        out = out + xt
        out = self.norm(out)
        out = out.reshape(B, J, T, H).permute(0, 2, 1, 3)  # back to (B, T, J, H)
        if DEBUG:
            print(f"TemporalAtt out: {out.shape}")
        return out


class STSABlock(nn.Module):

    def __init__(self, H, num_heads=NUM_HEADS, dropout=DROPOUT, ffn_mult=FFN_MULT):
        super().__init__()
        self.s_attn = SpatialAttentionBlock(H, num_heads=num_heads, dropout=dropout)
        self.t_attn = TemporalAttentionBlock(H, num_heads=num_heads, dropout=dropout)

        self.ffn_norm = nn.LayerNorm(H)
        self.ffn = nn.Sequential(
            nn.Linear(H, H * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H * ffn_mult, H),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(H)

    def forward(self, x):
        # x: (B, T, J, H)
        if DEBUG:
            print(f"[STSABlock] in: {x.shape}")

        # spatial (per-frame)
        s = self.s_attn(x)  # (B, T, J, H)
        x = x + s

        # temporal (per-joint)
        t = self.t_attn(x)  # (B, T, J, H)
        x = x + t

        # FFN per token
        B, T, J, H = x.shape
        y = x.reshape(B * T * J, H)
        y = self.ffn(y)
        y = y.reshape(B, T, J, H)
        x = x + y

        x = self.final_norm(x)
        if DEBUG:
            print(f"[STSABlock] out: {x.shape}")
        return x


class STSANet(nn.Module):
    def __init__(
        self,
        T=T,
        J=NUM_JOINTS,
        D=JOINT_DIM,
        H=H,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        ffn_mult=FFN_MULT,
        num_blocks=2,
    ):
        super().__init__()
        self.T = T
        self.J = J
        self.D = D
        self.H = H

        # embed per-joint coordinates to H
        self.embed = nn.Linear(D, H)

        # learnable positional embeddings
        # time pos: (1, T, 1, H), joint pos: (1, 1, J, H)
        self.pos_time = nn.Parameter(torch.randn(1, T, 1, H) * 0.02)
        self.pos_joint = nn.Parameter(torch.randn(1, 1, J, H) * 0.02)

        # stack of STSA blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                STSABlock(H, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult)
            )
        self.blocks = nn.ModuleList(blocks)

        self.pool_norm = nn.LayerNorm(H)
        self.cls = nn.Linear(H, num_classes)

    def forward(self, x):
        # x: (B, T, J, D)
        if DEBUG:
            print(f"[STSANet] in: {x.shape}")

        x = self.embed(x)  # (B, T, J, H)
        if DEBUG:
            print(f"[STSANet] after embed: {x.shape}")

        # add pos embeddings
        x = x + self.pos_time + self.pos_joint

        # blocks
        for idx, blk in enumerate(self.blocks):
            if DEBUG:
                print(f"[STSANet] Block {idx} input: {x.shape}")
            x = blk(x)

        # global average pool across time & joints
        x = self.pool_norm(x)
        pooled = x.mean(dim=(1, 2))  # (B, H)
        logits = self.cls(pooled)  # (B, C)
        if DEBUG:
            print(f"[STSANet] logits: {logits.shape}")
        return logits
