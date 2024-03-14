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
    
        #"""Pytorchify:
        f_loss = 0
        g_loss = 0
        n_loss = 0
        batch_size = deep_feats[0].shape[0]
        rho = []
        for i in range(batch_size):
            rho.append((torch.empty(1)).to(torch.device("cuda")))
            if w[i].numel() == 0: 
                # this sample has no overloaded instance
                continue
            else:
                #print(w[i].shape)
                #print(n[i].shape)
                #print(deep_feats[0][i].shape)
                #print((deep_feats[0][i] - n[i]).shape)
                rho[i] = -w[i] @ torch.linalg.norm(deep_feats[0][i] - n[i], dim = 1, keepdim = True)
                rho[i] = torch.exp(rho[i])
                rho[i] = rho[i] / torch.sum(rho[i])
        #print(rho)
        
        #print(cls_score)
        #print(target)
        for i in range(batch_size):
            if n[i].numel() != 0:
                f_loss += -w[i] @ torch.linalg.norm(deep_feats[0][i] - n[i], dim = 1, keepdim = True)
                print(torch.tensor([self.ce_loss(cls_score[i][j], target[i]) for j in range(cls_score[i].shape[0])]))
                print(rho[i])
                g_loss += rho[i] @ torch.tensor([self.ce_loss(cls_score[i][j], target[i]) for j in range(cls_score[i].shape[0])])
            else: 
                print(cls_score[i])
                print(target[i])
                n_loss += self.ce_loss(cls_score[i], target[i])
        loss = f_loss + g_loss


        """
        if len(n) == 0:
            loss = self.ce_loss(cls_score[0], target)

        else:
            f_loss = 0
            g_loss = 0
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
            """

        return loss
                



