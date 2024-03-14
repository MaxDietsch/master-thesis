import torch
import torch.nn as nn
from typing import Optional, List
from mmpretrain.structures import DataSample

from mmpretrain.registry import MODELS
from .linear_head import LinearClsHead
from ..losses.dos_loss import DOSLoss

@MODELS.register_module()
class DOSHead(LinearClsHead):
    """
    same like LinearClsHead but returns in updated loss function for the dosloss
    requirements:
        loss_module of the model should be DOSLoss
    """

    def __init__(self, **kwargs):
        super(DOSHead, self).__init__(**kwargs)
        
        assert isinstance(self.loss_module, DOSLoss), 'loss_module of the head should be DOSLoss when using DOSHead'


    def loss(self,
             deep_feats: torch.Tensor,
             n: List[torch.Tensor],
             w: torch.Tensor,
             data_samples: List[DataSample]) -> dict:

        """ Args:
            deep_feats (torch.Tensor): The deep feature tensor of the input tensor with shape
                (V,) in general.
            n (List[torch.Tensor]): The list containing all the nearest neighbours 
                of the input tensor 
            w (torch.Tensor): The list containing all the weights for the 
                input tensor 
            data_samples (List[DataSample], optional): The annotation
                data of every samples. """
        
        #"""Pytorchify
        cls_score = []
        for i, v_i in enumerate(n):
            if v_i.numel() == 0:
                # this sample has no overloaded instance
                cls_score.append(self(deep_feats[0][i].unsqueeze(dim=0)))
            else:
                print(v_i.shape)
                cls_score.append(self((v_i, )))
                print(self((v_i, )).shape)
        #"""
        """
        if len(n) == 0:
            cls_score.append(self(deep_feats))
        """


        target = torch.cat([i.gt_label for i in data_samples])

        losses = dict()
        loss = self.loss_module(deep_feats, cls_score, target, n, w)
        losses['loss'] = loss

        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses
