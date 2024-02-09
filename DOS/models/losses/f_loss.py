import torch 

import torch.nn as nn

class F_Loss(nn.Module):

    def __init__(self):
        super(F_Loss, self).__init__()

    def forward(self, deep_feat, n, w):
        loss = 0
        for idx, v_i in enumerate(n):
            loss += w[idx] * torch.linalg.norm(deep_feat, v_i)
        return loss
