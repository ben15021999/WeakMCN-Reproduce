# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeakREChead(nn.Module):
    def __init__(self, __C):
        super(WeakREChead, self).__init__()

    def forward(self, vis_fs, lan_fs):
        if self.training:
            return self.getContrast(vis_fs, lan_fs)
        else:
            return self.getPrediction(vis_fs, lan_fs)

    def getPrediction(self, vis_emb, lan_emb):
        vis_emb = F.normalize(vis_emb, dim=-1)
        lan_emb = F.normalize(lan_emb, dim=-1)
        sim_map = torch.einsum('bkd,byd->byk', vis_emb, lan_emb) # [B,1,K]
        _, best_idx = sim_map.max(dim=2, keepdim=True)
        predictions = torch.zeros_like(sim_map, dtype=torch.bool).scatter_(2, best_idx, True)
        return predictions

    def getContrast(self, vis_emb, lan_emb):
        vis_emb = F.normalize(vis_emb, dim=-1) # [B,K,D]
        lan_emb = F.normalize(lan_emb, dim=-1) # [B,1,D]
        B = vis_emb.size(0)

        sim_map = torch.einsum('avd,bqd->baqv', vis_emb, lan_emb) # [B,B,1,K]
        # top2 anchor trong mỗi cặp ảnh-câu
        max_sims, _ = sim_map.topk(k=2, dim=-1, largest=True, sorted=True) # [B,B,1,2]
        max_sims = max_sims.squeeze(2) # [B,B,2]

        max_sim_0 = max_sims[..., 0] # [B,B] best
        max_sim_1 = max_sims[..., 1] # [B,B] second

        # positive là đường chéo của best
        pos = torch.diagonal(max_sim_0, 0) # [B]

        # negative: bỏ đường chéo ở cả best và second, lấy tất cả cặp sai
        # mask off-diagonal
        eye = torch.eye(B, dtype=torch.bool, device=vis_emb.device)
        neg_0 = max_sim_0[~eye].view(B, B-1) # best của cặp sai
        neg_1 = max_sim_1[~eye].view(B, B-1) # second của cặp sai
        negatives = torch.cat([neg_0, neg_1], dim=1) # [B, 2*(B-1)]

        new_logits = torch.cat([pos.unsqueeze(1), negatives], dim=1) # [B, 1+2(B-1)]
        target = torch.zeros(B, dtype=torch.long, device=vis_emb.device) # positive ở cột 0

        ce_loss = F.cross_entropy(new_logits / 0.2, target) # temperature 0.2 cho ổn định

        # margin nội ảnh - chỉ tính trên cặp đúng
        pos_sim = sim_map[torch.arange(B), torch.arange(B), 0] # [B,K]
        top2 = pos_sim.topk(2, dim=1).values # [B,2]
        margin_loss = F.relu(0.1 - (top2[:,0] - top2[:,1])).mean() # margin 0.1 nhẹ hơn

        return ce_loss + 0.1 * margin_loss