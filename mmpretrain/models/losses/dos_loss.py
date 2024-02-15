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
        """ deep_feats (torch.Tensor): The deep feature vector of the input
            cls_score (torch.Tensor): The output vector of the input
            target (torch.Tensor): The target class of the input
            n (torch.Tensor): The deep features of the nearest neighbours of the input
            w (torch.Tensor): The weights for the given input
        """
        f_loss = 0
        g_loss = 0

        torch.autograd.set_detect_anomaly(True)

        # helper for g_loss
        rho = []
        

        for idy, w_i in enumerate(w):
            rho.append((torch.zeros(len(n))).to(torch.device("cuda")))
            for idx, v_i in enumerate(n):
                rho[idy][idx] += -w_i[idx] * torch.linalg.norm(deep_feats[0] - v_i)
            rho[idy] = torch.exp(rho[idy])
            rho[idy] = rho[idy] / torch.sum(rho[idy])



        for idy, w_i in enumerate(w):
            for idx, v_i in enumerate(n):
                f_loss += w_i[idx] * torch.linalg.norm(deep_feats[0] - v_i)
                g_loss += rho[idy][idx] * self.ce_loss(cls_score[idx].unsqueeze(0), target)

        loss = g_loss + f_loss
        return loss
                



