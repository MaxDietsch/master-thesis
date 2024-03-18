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
        """ 
            TODO: CHANGE COMMENTS
            deep_feats (torch.Tensor): The deep feature vector of the input
            cls_score (torch.Tensor): The output vector of the input
            target (torch.Tensor): The target class of the input
            n (torch.Tensor): The deep features of the nearest neighbours of the input
            w (torch.Tensor): The weights for the given input
        """
    
        #"""Pytorchify:
        loss = 0
        batch_size = deep_feats[0].shape[0]
        # calculate the rho values
        # print(torch.sum(deep_feats[0]))
        rho = []

        loss = 0
        # calculate the rho values
        # print(torch.sum(deep_feats[0]))
        rho = []

        rho.append((torch.empty(1)).to(torch.device("cuda")))
        if w.numel() != 0: 
            # this sample has an overloaded instance
            # deep_feats[0] is of shape 1 x feats, n is of shape k x feats -> norm: k x 1 -> transpose: 1 x k
            # w is of shape r x k -> rho is of shape r x k
            rho = -w * torch.linalg.norm(deep_feats[0] - n, dim = 1, keepdim = True).squeeze()
            rho = torch.exp(rho)
            rho = rho / torch.sum(rho, dim = 1, keepdim = True)

            print(w.shape)
            print((deep_feats[0] - n).shape)
            print(torch.linalg.norm(deep_feats[0] - n, ord = 2, dim = 1, keepdim = True).squeeze().shape)
            print(rho.shape)
            print(rho)
        #print(rho)
        
        if n.numel() != 0:
            # wi is of shape: r x k (for that class)
            # torch.linalg.norm(deep_feats[0][i] - n[i], ord = 2, dim = 1, keepdim = True).shape) shape k x 1
            # result is r x 1 (so for each weight vector) -> sum over it
            # implements: wi * ||f(x) - vi||**2 (sum for every i) , where wi is a component of 1 weight vector w and vi is 1 oversampled feature vector
            loss += torch.sum(w @ torch.linalg.norm(deep_feats[0] - n, dim = 1, keepdim = True))
            
            #print(w.shape)
            #print(torch.linalg.norm(deep_feats[0] - n, ord = 2, dim = 1, keepdim = True).shape)

            # cls_score is of shape k x classes, target is of shape 1 x 1
            # torch.tensor([self.ce_loss(cls_score[j], target) for j in range(cls_score[i].shape[0])]).shape is of shape k (loss for every oversampled examples)
            # rho is of shape r x k -> result will be r x 1 (for each weight vector) -> sum over it 
            # implements rho(vi, wi) * H(g(vi), y) (-> sum for every i), where g(vi) is prediction for oversamples feature and y is ground truth
            loss += torch.sum(rho @ torch.tensor([self.ce_loss(cls_score[j], target) for j in range(cls_score.shape[0])]).to(torch.device("cuda")))
            
            #print(cls_score.shape)
            #print(torch.tensor([self.ce_loss(cls_score[j], target) for j in range(cls_score.shape[0])]).shape)

        else:
            # for not oversampled instances take the normal loss
            loss += self.ce_loss(cls_score, target.unsqueeze(dim=0))
























        """with batch 
        for i in range(batch_size):
            rho.append((torch.empty(1)).to(torch.device("cuda")))
            if w[i].numel() == 0: 
                # this sample has no overloaded instance
                continue
            else:
                # deep_feats[0][i] is of shape k x feats, n[i] is of shape k x feats -> norm: k x 1 -> transpose: 1 x k
                # w[i] is of shape r x k -> rho[i] is of shape r x k
                rho[i] = -w[i] * torch.linalg.norm(deep_feats[0][i] - n[i], dim = 1, keepdim = True).squeeze()
                rho[i] = torch.exp(rho[i])
                rho[i] = rho[i] / torch.sum(rho[i], dim = 1, keepdim = True)

                #print(w[i].shape)
                #print((deep_feats[0][i] - n[i]).shape)
                #print(torch.linalg.norm(deep_feats[0][i] - n[i], ord = 2, dim = 1, keepdim = True).squeeze().shape)
                #print(rho[i].shape)
                #print(rho[i])
        #print(rho)
        
        for i in range(batch_size):
            if n[i].numel() != 0:
                # wi is of shape: r x k (for that class)
                # torch.linalg.norm(deep_feats[0][i] - n[i], ord = 2, dim = 1, keepdim = True).shape) shape k x 1
                # result is r x 1 (so for each weight vector) -> sum over it
                # implements: wi * ||f(x) - vi||**2 (sum for every i) , where wi is a component of 1 weight vector w and vi is 1 oversampled feature vector
                loss += torch.sum(w[i] @ torch.linalg.norm(deep_feats[0][i] - n[i], dim = 1, keepdim = True))
                
                #print(w[i].shape)
                #print(torch.linalg.norm(deep_feats[0][i] - n[i], ord = 2, dim = 1, keepdim = True).shape)

                # cls_score[i] is of shape k x classes, target is of shape 1 x 1
                # torch.tensor([self.ce_loss(cls_score[i][j], target[i]) for j in range(cls_score[i].shape[0])]).shape is of shape k (loss for every oversampled examples)
                # rho[i] is of shape r x k -> result will be r x 1 (for each weight vector) -> sum over it 
                # implements rho(vi, wi) * H(g(vi), y) (-> sum for every i), where g(vi) is prediction for oversamples feature and y is ground truth
                loss += torch.sum(rho[i] @ torch.tensor([self.ce_loss(cls_score[i][j], target[i]) for j in range(cls_score[i].shape[0])]).to(torch.device("cuda")))
                
                #print(cls_score[i].shape)
                #print(torch.tensor([self.ce_loss(cls_score[i][j], target[i]) for j in range(cls_score[i].shape[0])]).shape)

            else:
                # for not oversampled instances take the normal loss
                loss += self.ce_loss(cls_score[i], target[i].unsqueeze(dim=0))

        #print(f'n_loss: {n_loss}')
        #print(f'f_loss: {f_loss}')
        #print(f'g_loss: {g_loss}')
        """
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
            print(rho)
            
            #print(w)
            #print(n)
            #print(cls_score)
            for idy, w_i in enumerate(w):
                for idx, v_i in enumerate(n):
                    f_loss += w_i[idx] * torch.linalg.norm(deep_feats[0] - v_i)
                    g_loss += rho[idy][idx] * self.ce_loss(cls_score[idx].unsqueeze(0), target)
            
            loss = g_loss + f_loss
        #"""

        return loss
                



