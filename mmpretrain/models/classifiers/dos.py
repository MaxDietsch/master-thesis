
import torch
import torch.nn as nn
from typing import List, Optional
from mmpretrain.registry import MODELS
from .image import ImageClassifier
from ..heads.dos_head import DOSHead

@MODELS.register_module()
class DOSClassifier(ImageClassifier):

    def __init__(self, **kwargs):
        super(DOSClassifier, self).__init__(**kwargs)

        if not isinstance(self.head, DOSHead):
            raise TypeError('Head of the model should be of type DOSHead')

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):

        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples, n, w)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')


    def loss(self, input, data_samples, n, w):

        deep_feats = self.extract_feat(input)
        return self.head.loss(deep_feats, data_samples, n, w)
