import torch
import torch.nn as nn
from typing import List

from mmpretrain.registry import MODELS


@MODELS.register_module()
class DOSLoss(nn.Module):

    def __init__(self):
        super(DOSLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                deep_feats: torch.Tensor,
                cls_score: torch.Tensor, 
                target: torch.Tensor,
                n: List[torch.Tensor],
                w: List[torch.Tensor]) -> float:
        f_loss = 0
        g_loss = 0
        
        # helper for g_loss
        rho = torch.zeros(len(n))
        for idx, v_i in enumerate(n):
            rho[idx] = - w[idx] * torch.linalg.norm(deep_feats - v_i)
        rho = torch.exp(rho)
        rho = rho / torch.sum(rho)

        for w_i in w:
            for idx, v_i in enumerate(n):
                f_loss += w[idx] * torch.linalg.norm(deep_feats - v_i)
                g_loss += rho[idx] * self.ce_loss(cls_score, target)
        loss = g_loss + f_loss
        return loss
                



