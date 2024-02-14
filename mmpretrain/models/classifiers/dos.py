
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .image import ImageClassifier
from ..heads.dos_head import DOSHead

@MODELS.register_module()
class DOSClassifier(ImageClassifier):

    def __init__(self, **kwargs):
        super(DOSClassifier, self).__init__(**kwargs)

        if not isinstance(self.head, DOSHead):
            raise TypeError('Head of the model should be of type DOSHead')

    def loss(self, input, data_samples, n, w):

        deep_feats = self.extract_feat(input)
        return self.head.loss(deep_feats, data_samples, n, w)
