
import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict
from mmpretrain.structures import DataSample
from mmengine.optim import OptimWrapper

from mmpretrain.registry import MODELS
from .image import ImageClassifier
from ..heads.dos_head import DOSHead

@MODELS.register_module()
class DOSClassifier(ImageClassifier):
    """Dos Classifier is exactly like ImageClassifier, but things changed: 
        give the n and w parameters of DOS to the loss function of the head
        require DOSHead as head of the model"""

    def __init__(self, **kwargs):
        super(DOSClassifier, self).__init__(**kwargs)

        if not isinstance(self.head, DOSHead):
            raise TypeError('Head of the model should be of type DOSHead')



    def train_step(self, data: Union[dict, tuple, list],
                   n: List[torch.Tensor],
                   w: torch.Tensor, 
                   optim_wrapper: OptimWrapper
                   ) -> Dict[str, torch.Tensor]:
       

        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, n = n, w = w, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars


    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str,
                     n: List[torch.Tensor] = None,
                     w: torch.Tensor = None) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, n=n, w=w, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, n=n, w=w, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results



    def forward(self,
                inputs: torch.Tensor,
                n: List[torch.Tensor] = None,
                w: torch.Tensor = None,
                data_samples: Optional[List[DataSample]] = None, 
                mode: str = 'tensor'):
        """ Args:
            inputs (torch.Tensor): The input tensor with shape
                (V, ) in general.
            n (List[torch.Tensor]): The list containing all the nearest neighbours 
                of the input tensor 
            w torch.Tensor: The tensor containing all the weights for the 
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

        deep_feats = self.extract_feat(inputs)
        return self.head.loss(deep_feats, n, w, data_samples)
