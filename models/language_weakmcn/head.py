# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeakREChead(nn.Module):
    def __init__(self, __C):
        super(WeakREChead, self).__init__()

    def forward(self, vis_fs, lan_fs):
        if self.training:
            loss = self.getContrast(vis_fs, lan_fs)
            return loss
        else:
            predictions = self.getPrediction(vis_fs, lan_fs)
            return predictions

    def getPrediction(self, vis_emb, lan_emb):
        # vis_emb: [B, K, D], lan_emb: [B, Q, D]
        sim_map = torch.einsum(
            'bkd, bqd -> bqk', vis_emb, lan_emb)  # [B, Q, K]
        # chọn query tốt nhất cho mỗi câu
        maxval, v = sim_map.max(dim=-1, keepdim=True)  # v: [B, Q, 1]
        # trả về dạng mask [B, Q, K] như code cũ
        predictions = torch.zeros_like(sim_map).scatter(-1, v, 1).bool()
        return predictions

    def getContrast(self, vis_emb, lan_emb):
        # vis_emb: [B, K, D], lan_emb: [B, Q, D]
        B, K, D = vis_emb.shape
        Q = lan_emb.shape[1]

        # 1. sim trong chính ảnh: [B, Q, K]
        sim_pos = torch.einsum('bkd, bqd -> bqk', vis_emb, lan_emb)

        # 2. QueryMatch: lấy query tốt nhất làm pseudo label
        with torch.no_grad():
            pseudo_label = sim_pos.argmax(dim=-1)  # [B, Q]

        # loss trong ảnh: ép query được chọn cao hơn 99 query còn lại
        # tau=1.0 chính là softmax thường như bạn nói
        loss_intra = F.cross_entropy(
            sim_pos.reshape(B*Q, K),
            pseudo_label.reshape(B*Q)
        )

        # 3. loss giữa các ảnh: ép câu của ảnh này phải giống query ảnh này hơn ảnh khác
        # [Ba, Bv, Q, K] -> max trên K -> [Ba, Bv, Q] -> mean Q -> [Ba, Bv]
        sim_all = torch.einsum('avd, bqd -> baqv', vis_emb, lan_emb)
        sim_all_max = sim_all.max(dim=-1).values.mean(dim=-1)  # [B, B]

        target = torch.arange(B).to(vis_emb.device)
        loss_inter = (F.cross_entropy(sim_all_max, target) +
                      F.cross_entropy(sim_all_max.t(), target)) / 2.0

        # tổng = trong ảnh + giữa ảnh
        return loss_intra + 0.5 * loss_inter

    # def getPrediction(self, vis_emb, lan_emb):
    #     sim_map = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)
    #     maxval, v = sim_map.max(dim=2, keepdim=True)
    #     predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2, v.expand(sim_map.shape), 1).bool()
    #     return predictions

    # def getContrast(self, vis_emb, lan_emb):
    #     sim_map = torch.einsum('avd, bqd -> baqv', vis_emb, lan_emb)
    #     batchsize = sim_map.shape[0]
    #     max_sims, _ = sim_map.topk(k=2, dim=-1, largest=True, sorted=True)
    #     max_sims = max_sims.squeeze(2)

    #     # Negative Anchor Augmentation
    #     max_sim_0, max_sim_1 = max_sims[..., 0], max_sims[..., 1]
    #     max_sim_1 = max_sim_1.masked_select(~torch.eye(batchsize).bool().to(max_sim_1.device)).contiguous().view(
    #         batchsize,
    #         batchsize - 1)
    #     new_logits = torch.cat([max_sim_0, max_sim_1], dim=1)

    #     target = torch.eye(batchsize).to(vis_emb.device)
    #     target_pred = torch.argmax(target, dim=1)
    #     loss = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)
    #     return loss
