
import torch
import torch.nn as nn
from typing import List, Optional
from mmpretrain.structures import DataSample

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
                n: List[torch.Tensor],
                w: List[torch.Tensor],
                data_samples: Optional[List[DataSample]] = None, 
                mode: str = 'tensor'):
        """ Args:
            inputs (torch.Tensor): The input tensor with shape
                (V, ) in general.
            n (List[torch.Tensor]): The list containing all the nearest neighbours 
                of the input tensor 
            w (List[torch.Tensor]): The list containing all the weights for the 
                input tensor 
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'."""

        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, n, w, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')


    def loss(self, 
             inputs: torch.Tensor,
             n: List[torch.Tensor],
             w: List[torch.Tensor],
             data_samples: Optional[List[DataSample]] = None) -> dict:
        """ Args:
            inputs (torch.Tensor): The input tensor with shape
                (V, ) in general.
            n (List[torch.Tensor]): The list containing all the nearest neighbours 
                of the input tensor 
            w (List[torch.Tensor]): The list containing all the weights for the 
                input tensor 
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None."""

        deep_feats = self.extract_feat(input)
        return self.head.loss(deep_feats, data_samples, n, w)
