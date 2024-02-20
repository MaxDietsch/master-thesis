import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS


# log_softmax version as it tends to be more stable 
def log_softmax(cls_score, label, xi): 
    
    """ Want to do this, but with list comprehension
    l = torch.zeros(pred.shape)
    for i, lab in enumerate(label): 
        m = xi[lab, torch.argmax(pred[i])]
        l[i] = m.log() + pred - (m + pred.exp()).sum(-1).log().unsqueeze(-1)
    """
    log_s = torch.stack([xi[lab, torch.argmax(cls_score[i])].log() + cls_score[i] - (xi[lab, torch.argmax(cls_score[i])] * cls_score[i].exp()).sum().log() for i, lab in enumerate(label)])
    return log_s

# negative log_likelihood, used because we use log_softmax
def negative_log_likelihood(log_s, label):
    return -log_s[range(label.shape[0]), label].mean()



@MODELS.register_module()
class CoSenCrossEntropyLoss(nn.Module):

    def __init__(self, num_classes, learning_rate):
        """
        Args:   num_classes (int): The number of classes.
                learning_rate (float): The learning rate for xi
        """

        super(CoSenCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.xi = torch.ones((num_classes, num_classes))


    def forward(self, cls_score, label):
        """
        Args:   cls_score (torch.Tensor): The prediction scores with shape(Batch, #Classes)
                label (torch.Tensor): The label with shape (Batch, )
                xi (torch.Tensor): The co-sen matrix with shape (#Classes, #Classes)
        """
        log_s = log_softmax(cls_score, label, self.xi)
        nll_loss = negative_log_likelihood(log_s, label)

        return nll_loss

    def compute_grad(self, v_a):
        """
        Args:   v_a (torch.Tensor): The vectorized T tensor, like in the paper.
                v_b (torch.Tensor): The vectorized xi tensor, like in the paper.
        """
        v_b = self.xi.view(-1, 1)
        return -(v_a - v_b) * torch.ones((self.num_classes * self.num_classes, 1))

    def update_xi(self, v_a):
        """
        Args:   v_a (torch.Tensor): The vectorized T tensor, like in the paper.
                v_b (torch.Tensor): The vectorized xi tensor, like in the paper.
        """

        self.xi -= self.learning_rate * self.compute_grad(v_a).view(self.xi.shape)

    def set_xi_lr(self, new_lr):
        """
        Args:   new_lr (float): The new learning rate for the xi learning rate
        """

        self.learning_rate = new_lr


