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

        print(w) 
        print(w[0])
        print(w[0][0, 0])
        # helper for g_loss
        rho = torch.zeros(len(n))
        for idx, v_i in enumerate(n):
            for idy, w_i in enumerate(w[0]):
                print(w[idx][idy])
                print(w_i)
                rho[idx] = - w[idx][idy] * torch.linalg.norm(deep_feats[0] - v_i)
        rho = torch.exp(rho)
        rho = rho / torch.sum(rho)

        for w_i in w:
            for idx, v_i in enumerate(n):
                f_loss += w[idx] * torch.linalg.norm(deep_feats[0] - v_i)
                g_loss += rho[idx] * self.ce_loss(cls_score, target)
        loss = g_loss + f_loss
        return loss
                



