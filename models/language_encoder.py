# coding=utf-8

import torch
import torch.nn as nn

from transformers import (
    CLIPTokenizer,
    CLIPTextModel
)

from models.network_blocks import SA,AttFlat
from utils.utils import make_mask


class LSTM_SA(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(LSTM_SA, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.GRU(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            dropout=__C.DROPOUT_R,
            bidirectional=False
        )

        self.sa_list = nn.ModuleList([SA(__C) for _ in range(__C.N_SA)])
        self.att_flat=AttFlat(__C)
        if __C.EMBED_FREEZE:
            self.frozen(self.embedding)
    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    def forward(self, ques_ix):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)


        for sa in self.sa_list:
            lang_feat = sa(lang_feat, lang_feat_mask)

        flat_lang_feat = self.att_flat(lang_feat, lang_feat_mask)
        return  {
            'flat_lang_feat':flat_lang_feat,
            'lang_feat':lang_feat,
            'lang_feat_mask':lang_feat_mask
        }


class CLIP_SA(nn.Module):
    def __init__(self, __C):
        super(CLIP_SA, self).__init__()

        # =====================================
        # Load CLIP from HuggingFace
        # =====================================
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.clip_text = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        clip_dim = self.clip_text.config.hidden_size
        # ViT-B/32 => 512

        # =====================================
        # Optional projection
        # =====================================
        self.proj = nn.Linear(
            clip_dim,
            __C.HIDDEN_SIZE
        )

        # =====================================
        # Self Attention blocks
        # =====================================
        self.sa_list = nn.ModuleList(
            [SA(__C) for _ in range(__C.N_SA)]
        )

        self.att_flat = AttFlat(__C)

        # freeze CLIP nếu muốn
        if __C.EMBED_FREEZE:
            for p in self.clip_text.parameters():
                p.requires_grad = False

    def forward(self, text):

        # =====================================
        # Tokenize
        # =====================================
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].to(
            next(self.parameters()).device
        )

        attention_mask = tokens["attention_mask"].to(
            next(self.parameters()).device
        )

        # =====================================
        # CLIP Text Encoder
        # =====================================
        outputs = self.clip_text(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [B, L, 512]
        lang_feat = outputs.last_hidden_state

        # project nếu cần
        lang_feat = self.proj(lang_feat)

        # =====================================
        # mask
        # True = masked position
        # =====================================
        lang_feat_mask = (
            attention_mask == 0
        ).unsqueeze(1).unsqueeze(2)

        # =====================================
        # SA blocks
        # =====================================
        for sa in self.sa_list:
            lang_feat = sa(
                lang_feat,
                lang_feat_mask
            )

        # =====================================
        # AttFlat
        # =====================================
        flat_lang_feat = self.att_flat(
            lang_feat,
            lang_feat_mask
        )

        return {
            'flat_lang_feat': flat_lang_feat,
            'lang_feat': lang_feat,
            'lang_feat_mask': lang_feat_mask
        }


backbone_dict={
    'lstm': LSTM_SA,
    'clip': CLIP_SA,
}

def language_encoder(__C, pretrained_emb, token_size):
    lang_enc=backbone_dict[__C.LANG_ENC](__C, pretrained_emb, token_size)
    return lang_enc