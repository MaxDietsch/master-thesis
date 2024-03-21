import torch
import torch.nn as nn
from typing import Optional, List
from mmpretrain.structures import DataSample

from mmpretrain.registry import MODELS
from .linear_head import LinearClsHead
from ..losses.bmu_loss import BMULoss

@MODELS.register_module()
class BMUHead(LinearClsHead):
    """
    same like LinearClsHead but altered loss function (module), as the data sample is different 
    in BlancedMixUP strategy. This is checked in the init function
    requirements:
        loss_module of the model should be BMULoss
    """

    def __init__(self, **kwargs):
        super(BMUHead, self).__init__(**kwargs)
        
        assert isinstance(self.loss_module, BMULoss), 'loss_module of the head should be BMULoss when using DOSHead'
    
    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""

        one_hot_vecs = torch.stack([i for i in data_samples])
        
        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

