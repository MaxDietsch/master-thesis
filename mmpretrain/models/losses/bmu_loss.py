import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpretrain.registry import MODELS

@MODELS.register_module()
class BMULoss(nn.Module):
    """Cross Entropy Loss but input is cls_score (tensor) and the label as mixed encoded vector 
        This is the difference to CrossEntropyLoss function

    """

    def __init__(self):
        super(BMULoss, self).__init__()

    def forward(self, cls_score, one_hot_label):

        preds = F.softmax(cls_score, dim = 1)
        one_hot_label.to(torch.device('cuda'))

        print(one_hot_label.device)
        print(preds.device)


        loss = -one_hot_label * torch.log(preds + 1e-9)
        print(loss.shape)

        return loss.mean(1).mean(0)




