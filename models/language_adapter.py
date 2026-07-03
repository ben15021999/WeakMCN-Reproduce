import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Feed Forward Network
# ---------------------------------------------------------
class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Multi-head Cross Attention
# ---------------------------------------------------------
class CrossAttention(nn.Module):

    def __init__(
            self,
            visual_dim,
            lang_dim,
            num_heads=8,
            dropout=0.1
    ):
        super().__init__()

        assert visual_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads

        self.q_proj = nn.Linear(visual_dim, visual_dim)
        self.k_proj = nn.Linear(lang_dim, visual_dim)
        self.v_proj = nn.Linear(lang_dim, visual_dim)

        self.out_proj = nn.Linear(visual_dim, visual_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            visual,
            language,
            language_mask=None
    ):
        """
        visual:
            (B,N,C)

        language:
            (B,L,D)

        language_mask:
            (B,1,1,L)
        """

        B, N, C = visual.shape
        L = language.shape[1]

        q = self.q_proj(visual)
        k = self.k_proj(language)
        v = self.v_proj(language)

        q = q.view(B, N, self.num_heads,
                   self.head_dim).transpose(1, 2)

        k = k.view(B, L, self.num_heads,
                   self.head_dim).transpose(1, 2)

        v = v.view(B, L, self.num_heads,
                   self.head_dim).transpose(1, 2)

        score = torch.matmul(
            q,
            k.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)

        if language_mask is not None:

            if language_mask.dim() == 4:
                mask = language_mask.squeeze(1).squeeze(1)

            elif language_mask.dim() == 2:
                mask = language_mask

            else:
                mask = language_mask

            score = score.masked_fill(mask[:, None, None, :], -1e9)

        attn = F.softmax(score, dim=-1)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.transpose(
            1,
            2
        ).reshape(B, N, C)

        out = self.out_proj(out)

        return out

# ---------------------------------------------------------
# Spatial Semantic Gate
# ---------------------------------------------------------


class SpatialGate(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        """
        x:
            (B,N,C)
        """
        score = self.gate(x)
        score = torch.sigmoid(score)
        return x * score
    
# ---------------------------------------------------------
# Language Guided Semantic Adapter
# ---------------------------------------------------------


class LanguageGuidedAdapter(nn.Module):

    def __init__(
            self,
            visual_dim,
            lang_dim,
            num_heads=8,
            dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(visual_dim)

        self.cross_attn = CrossAttention(
            visual_dim,
            lang_dim,
            num_heads,
            dropout
        )

        self.norm2 = nn.LayerNorm(visual_dim)

        self.ffn = FeedForward(
            visual_dim,
            visual_dim * 4,
            dropout
        )

        self.spatial_gate = SpatialGate(
            visual_dim
        )

    def forward(
            self,
            visual_feature,
            language_feature,
            language_mask=None
    ):
        """
        visual_feature:
            (B,C,H,W)

        language_feature:
            (B,L,D)
        """

        B, C, H, W = visual_feature.shape

        visual = visual_feature.flatten(
            2
        ).transpose(
            1,
            2
        )

        residual = visual

        visual = self.norm1(visual)

        semantic = self.cross_attn(
            visual,
            language_feature,
            language_mask
        )

        visual = residual + semantic
        residual = visual
        visual = self.norm2(visual)
        visual = residual + self.ffn(visual)
        visual = self.spatial_gate(visual)
        visual = visual.transpose(
            1,
            2
        ).reshape(
            B,
            C,
            H,
            W
        )

        return visual
