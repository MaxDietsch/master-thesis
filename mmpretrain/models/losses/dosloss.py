import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from .utils import weight_reduce_loss

import numpy as np



@MODELS.register.module()
class DOSLoss(nn.Module):

    def __init__(self,
                 overload_param):
        super(DOSLoss, self).__init__()
        self.overload_param = overload_param
       
    def forward (self, cls_score, label, deep_features, neighbors):

        # micro-cluster loss functions for the weighted instances
        l_f = 0
        weights_vectors = [[] * len(cls_score)]
        r = 0
        for i in range(len(cls_score)):
            for j in range(r):
                w = np.random.rand(self.overload_param)
                w /= np.sum(w)
                weights_vectors[i].append(w)
        for j in range(len(cls_score)):
            for i in range(self.overload_param):
                lf += weights_vectors[j][i] * np.sqrt(np.sum((deep_features[j] - neighbors[j][i])**2))




