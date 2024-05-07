import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS


# log_softmax version with cost matrix as it tends to be more stable 
def log_softmax(cls_score, label, xi): 
    
    print(xi[0, : ])
    print(cls_score[0])
    print(xi[0, : ] * cls_score[0])

    log_s = torch.stack([xi[lab, torch.argmax(cls_score[i])].log() + cls_score[i] - (xi[lab, : ] * cls_score[i].exp()).sum().log() for i, lab in enumerate(label)])
    return log_s

# negative log_likelihood to compute loss of batch, used because we use log_softmax
def negative_log_likelihood(log_s, label):
    return -log_s[range(label.shape[0]), label].mean()



@MODELS.register_module()
class CoSenCrossEntropyLoss(nn.Module):
    """ Loss for the paper: Cost-Sensitive Learning of Deep Feature Representations from Imbalanced Data
        it uses a cost matrix xi do modify the ce loss. 

        Args:   num_classes (int): The number of classes.
                learning_rate (float): The learning rate for xi
    """

    def __init__(self, num_classes, learning_rate):

        super(CoSenCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.xi = torch.ones((num_classes, num_classes), device = torch.device('cuda'))


    def forward(self, cls_score, label):
        """
        Args:   cls_score (torch.Tensor): The prediction scores with shape(Batch, #Classes)
                label (torch.Tensor): The label with shape (Batch, )
        """
        log_s = log_softmax(cls_score, label, self.xi)
        nll_loss = negative_log_likelihood(log_s, label)

        return nll_loss

    def compute_grad(self, v_a):
        """
        Args:   v_a (torch.Tensor): The vectorized T tensor, like in the paper.
        """
        v_b = self.xi.view(-1, 1).to(torch.device('cuda'))
        return -(v_a - v_b) * torch.ones((self.num_classes * self.num_classes, 1), device = torch.device('cuda'))

    def update_xi(self, v_a):
        """
        Args:   v_a (torch.Tensor): The vectorized T tensor, like in the paper.
        """

        self.xi -= self.learning_rate * self.compute_grad(v_a).view(self.xi.shape)

    def set_xi_lr(self, new_lr):
        """
        Args:   new_lr (float): The new learning rate for the xi learning rate
        """

        self.learning_rate = new_lr

    def get_xi_lr(self):
        return self.learning_rate


