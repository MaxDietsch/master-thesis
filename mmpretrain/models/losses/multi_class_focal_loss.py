# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmpretrain.registry import MODELS
from .utils import convert_to_one_hot, weight_reduce_loss


def softmax_focal_loss(pred,
                       target,
                       gamma=2.0,
                       alpha=0.25,
                       ):
    r"""ce focal loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*).
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha (float): A balanced form for Focal Loss. Defaults to 0.25.

    Returns:
        torch.Tensor: Loss.
    """
    pred_probs = F.softmax(pred, dim = 1)
    target = target.type_as(pred)
    print(pred_probs)
    print(target)
    row_indices = torch.arange(pred_probs.shape[0])
    pt = pred_probs[ row_indices , target.int()]
    print(pt)
    focal_weight = alpha * (1 - pt).pow(gamma)
    print(focal_weight)
    print(-focal_weight * torch.log(pt))
    loss = (-focal_weight * torch.log(pt)).sum()
    print(loss)
    return loss


@MODELS.register_module()
class MultiClassFocalLoss(nn.Module):
    """Focal loss.

    Args:
        gamma (float): Focusing parameter in focal loss.
            Defaults to 2.0.
        alpha (float): The parameter in balanced form of focal
            loss. Defaults to 0.25.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25):

        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        r"""Sigmoid focal loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*), N or (N,1).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        loss_cls = softmax_focal_loss(
            pred,
            target,
            gamma=self.gamma,
            alpha=self.alpha)
        return loss_cls
