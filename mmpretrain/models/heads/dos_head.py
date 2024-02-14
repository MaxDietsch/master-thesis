import torch
import torch.nn as nn
from typing import Optional, Tuple

from mmpretrain.registry import MODELS
from .linear_head import LinearClsHead
from ..losses.dos_loss import DOSLoss

@MODELS.register_module()
class DOSHead(LinearClsHead):

    def __init__(self, **kwargs):
        super(DOSHead, self).__init__(**kwargs)
        
        if not isinstance(self.loss_module, DOSLoss):
            raise TypeError('Loss of the model should be of type DOSLoss')



    def loss(self, deep_feats, data_samples, n, w):

        cls_score = self(deep_feats)

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
