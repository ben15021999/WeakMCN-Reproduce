from models.network_blocks import SA, AttFlat
from utils.utils import make_mask
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import CLIPTextModelWithProjection as CLIPTP
from transformers import (AutoTokenizer, AutoProcessor,
                          AutoModel, CLIPTextConfig)


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
        self.att_flat = AttFlat(__C)
        if __C.EMBED_FREEZE:
            self.frozen(self.embedding)

    def frozen(self, module):
        module.eval()
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
        return {
            'flat_lang_feat': flat_lang_feat,
            'lang_feat': lang_feat,
            'lang_feat_mask': lang_feat_mask
        }


class BERT_SA(nn.Module):
    def __init__(self, __C):
        super(BERT_SA, self).__init__()
        # Khởi tạo backbone BERT
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

        # BERT base có hidden_size = 768
        bert_out_dim = self.bert.config.hidden_size

        # Nếu HIDDEN_SIZE của network khác 768, chiếu về HIDDEN_SIZE
        if bert_out_dim != __C.HIDDEN_SIZE:
            self.proj = nn.Linear(bert_out_dim, __C.HIDDEN_SIZE)
        else:
            self.proj = nn.Identity()

        self.sa_list = nn.ModuleList([SA(__C) for _ in range(__C.N_SA)])
        self.att_flat = AttFlat(__C)

        # Đóng băng BERT nếu được cấu hình
        if __C.EMBED_FREEZE:
            self.frozen(self.bert)

    def frozen(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, y):
        input_ids = y['input_ids']
        attention_mask = y['attention_mask']
        # 1. Trích xuất đặc trưng từ BERT
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        # last_hidden_state có kích thước: (batch_size, seq_len, 768)
        lang_feat = bert_output.last_hidden_state

        # 2. Chiếu đặc trưng về kích thước HIDDEN_SIZE nếu cần
        lang_feat = self.proj(lang_feat)

        # 3. Tạo mask cho khối Self-Attention / AttFlat
        # Chuyển attention_mask (batch, seq_len) -> (batch, 1, 1, seq_len) hoặc tương đương hàm make_mask
        lang_feat_mask = make_mask(input_ids.unsqueeze(2))

        # 4. Đưa qua các lớp SA và AttFlat hiện tại
        for sa in self.sa_list:
            lang_feat = sa(lang_feat, lang_feat_mask)

        flat_lang_feat = self.att_flat(lang_feat, lang_feat_mask)

        return {
            'flat_lang_feat': flat_lang_feat,
            'lang_feat': lang_feat,
            'lang_feat_mask': lang_feat_mask
        }


backbone_dict = {
    'lstm': LSTM_SA,
    'bert': BERT_SA,
}


def language_encoder(__C, pretrained_emb=None, token_size=None):
    if __C.LANG_ENC == 'lstm':
        lang_enc = backbone_dict[__C.LANG_ENC](__C, pretrained_emb, token_size)
    else:
        lang_enc = backbone_dict[__C.LANG_ENC](__C)
    return lang_enc
