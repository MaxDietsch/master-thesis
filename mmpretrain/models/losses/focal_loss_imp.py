# same as focal_loss.py but the alpha balnces all classes and not only binary
# and cross entropy is used instead of bce 
import torch.nn as nn
import torch.nn.functional as F
import torch

from mmpretrain.registry import MODELS
from .utils import convert_to_one_hot, weight_reduce_loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=torch.tensor([1, 1, 1, 1]),
                       reduction='mean',
                       avg_factor=None):
    r"""Sigmoid focal loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Defaults to None.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha (torch tensor (size = number of classes)): A balanced form for Focal Loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' ,
            loss is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1  - target)) * pt.pow(gamma)

    loss = F.cross_entropy(pred, target, reduction = 'none') * focal_weight

    # is the alpha necessary ? 
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss


def softmax_focal_loss( pred,
                target,
                weight=None,
                gamma=2.0,
                alpha=torch.tensor([1, 1, 1, 1]),
                reduction='mean',
                avg_factor=None):
    r"""focal loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Defaults to None.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha (torch tensor (dimension = number of classes)): A balanced form for Focal Loss. Defaults to [1, 1, 1, 1].
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' ,
            loss is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    target = target.type_as(pred)
    pred_prob = F.softmax(pred, dim = 1)
    pt = (1 - pred_prob) * target + pred_prob * (1 - target)
    focal_weight = alpha * pt.pow(gamma)

    loss = F.cross_entropy(pred, target, reduction = 'none') * focal_weight

    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



@MODELS.register_module()
class FocalLossImp(nn.Module):
    """Focal loss.

    Args:
        gamma (float): Focusing parameter in focal loss.
            Defaults to 2.0.
        alpha (array of float etc.): The parameter to balance loss of each class.
            Defaults to [1, 1, 1, 1] (perfect balance)
        reduction (str): The method used to reduce the loss into
            a scalar. Options are "none" and "mean". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.

        act_fct ("softmax" or "sigmoid") defines if sigmoid or softmax activation is 
            used to normalize the logits. 
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=[0.25, 0.25, 0.25, 0.25],
                 reduction='mean',
                 loss_weight=1.0, 
                 act_fct="softmax"):

        super(FocalLossImp, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.act_fct = act_fct

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        r"""focal loss.

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
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1):
            target = convert_to_one_hot(target.view(-1, 1), pred.shape[-1])
        
        if self.act_fct == "softmax":
            loss_cls = self.loss_weight * softmax_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        
        elif self.act_fct == "sigmoid":
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        return loss_cls
