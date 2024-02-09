import torch 
import torch.nn as nn
import torch.nn.functional as F


class G_Loss(nn.Module):

    def __init__(self):
        super(G_Loss, self).__init__()


    def forward(deep_feat, classification, label, n, w):
        loss = 0
        
        # help for normalizer
        rho = torch.zeros(len(n))
        for idx, v_i in enumerate(n):
            rho.append(- w[idx] * torch.linalg.norm(deep_feat, n))
        rho = torch.exp(rho)
        rho = rho / torch.sum(rho)


        for idx, v_i in enumerate(n):
            loss += rho[idx] * nn.CrossEntropyLoss(F.softmax(classification), F.one_hot(label, 4))
            return loss

