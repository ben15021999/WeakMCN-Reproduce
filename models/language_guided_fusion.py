import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, dim, ratio=4):

        super().__init__()

        hidden = dim * ratio

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):

        return self.net(x)


class LanguageCrossAttention(nn.Module):

    def __init__(
            self,
            visual_dim,
            language_dim,
            num_heads=8):

        super().__init__()

        self.visual_dim = visual_dim

        self.num_heads = num_heads

        self.head_dim = visual_dim // num_heads

        assert visual_dim % num_heads == 0

        self.q_proj = nn.Linear(
            visual_dim,
            visual_dim
        )

        self.k_proj = nn.Linear(
            language_dim,
            visual_dim
        )

        self.v_proj = nn.Linear(
            language_dim,
            visual_dim
        )

        self.out_proj = nn.Linear(
            visual_dim,
            visual_dim
        )

        self.norm1 = nn.LayerNorm(visual_dim)

        self.norm2 = nn.LayerNorm(visual_dim)

        self.ffn = FeedForward(visual_dim)

    def forward(
            self,
            visual,
            language,
            language_mask=None):
        """
        visual:

        B,C,H,W

        language

        B,L,C
        """

        B, C, H, W = visual.shape

        N = H*W

        visual = visual.flatten(2).transpose(1, 2)

        Q = self.q_proj(visual)

        K = self.k_proj(language)

        V = self.v_proj(language)

        Q = Q.view(
            B,
            N,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        score = torch.matmul(
            Q,
            K.transpose(-1, -2)
        )

        score /= self.head_dim**0.5

        if language_mask is not None:

            mask = language_mask.squeeze(1).squeeze(1)

            score = score.masked_fill(
                mask[:, None, None, :],
                -1e9
            )

        attn = F.softmax(
            score,
            dim=-1
        )

        out = torch.matmul(
            attn,
            V
        )

        out = out.transpose(
            1,
            2
        ).reshape(
            B,
            N,
            C
        )

        out = self.out_proj(out)

        out = self.norm1(
            visual + out
        )

        out = self.norm2(
            out + self.ffn(out)
        )

        out = out.transpose(
            1,
            2
        ).reshape(
            B,
            C,
            H,
            W
        )

        return out
    

class VisualCrossAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8):

        super().__init__()

        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)

        self.k_proj = nn.Linear(dim, dim)

        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = FeedForward(dim)

    def forward(
            self,
            query_feature,
            memory_feature):

        B, C, H, W = query_feature.shape

        N = H*W

        query = query_feature.flatten(
            2
        ).transpose(
            1,
            2
        )

        memory = memory_feature.flatten(
            2
        ).transpose(
            1,
            2
        )

        Q = self.q_proj(query)

        K = self.k_proj(memory)

        V = self.v_proj(memory)

        Q = Q.view(
            B, N, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        score = torch.matmul(
            Q,
            K.transpose(-1, -2)
        )

        score /= self.head_dim**0.5

        attn = F.softmax(
            score,
            dim=-1
        )

        out = torch.matmul(
            attn,
            V
        )

        out = out.transpose(
            1,
            2
        ).reshape(
            B,
            N,
            C
        )

        out = self.out_proj(out)

        out = self.norm1(
            query + out
        )

        out = self.norm2(
            out+self.ffn(out)
        )

        out = out.transpose(
            1,
            2
        ).reshape(
            B,
            C,
            H,
            W
        )

        return out
    

class AdaptiveFusionGate(nn.Module):

    def __init__(
            self,
            dim):

        super().__init__()

        self.gate = nn.Sequential(

            nn.Conv2d(
                dim*3,
                dim,
                1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                dim,
                3,
                1
            )

        )

    def forward(
            self,
            yolo,
            dino,
            sam):

        weight = self.gate(

            torch.cat(
                [
                    yolo,
                    dino,
                    sam
                ],
                dim=1
            )
        )

        weight = F.softmax(
            weight,
            dim=1
        )

        out = (

            weight[:, 0:1]*yolo +

            weight[:, 1:2]*dino +

            weight[:, 2:3]*sam

        )

        return out


class CrossGatedInteraction(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.dino_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        self.sam_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, dino, sam):

        x = torch.cat([dino, sam], dim=1)

        gate_dino = self.dino_gate(x)

        gate_sam = self.sam_gate(x)

        dino = dino * gate_dino

        sam = sam * gate_sam

        return dino, sam

class LanguageGuidedFusion(nn.Module):

    def __init__(
            self,
            dim,
            language_dim):

        super().__init__()

        self.language_attention = LanguageCrossAttention(
            dim,
            language_dim
        )

        self.dino_attention = VisualCrossAttention(
            dim
        )

        self.sam_attention = VisualCrossAttention(
            dim
        )

        self.gate = AdaptiveFusionGate(
            dim
        )

        self.interaction = CrossGatedInteraction(dim)

    def forward(
            self,
            yolo,
            dino,
            sam,
            lang_feat,
            lang_mask):

        language_feature = self.language_attention(
            yolo,
            lang_feat,
            lang_mask
        )

        dino_feature = self.dino_attention(
            language_feature,
            dino
        )

        sam_feature = self.sam_attention(
            language_feature,
            sam
        )

        dino_feature, sam_feature = self.interaction(
            dino_feature,
            sam_feature
        )

        out = self.gate(
            language_feature,
            dino_feature,
            sam_feature
        )

        return out
