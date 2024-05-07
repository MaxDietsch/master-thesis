import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict
from mmpretrain.structures import DataSample
from mmengine.optim import OptimWrapper

from mmpretrain.registry import MODELS
from .image import ImageClassifier
from ..heads.cosen_linear_head import CoSenLinearClsHead

@MODELS.register_module()
class CoSenClassifier(ImageClassifier):
    """ 
        for implementation of: Cost-Sensitive Learning of Deep Feature Representations from Imbalanced Data
        Cosen Classifier is exactly like ImageClassifier, but things changed:
        handle in the train_step function, that the loss now return the loss and the predicted labels
        
        REQUIRES:
            CosenClsHead as head of the model

    """

    def __init__(self, **kwargs):
        super(CoSenClassifier, self).__init__(**kwargs)

        if not isinstance(self.head, CoSenLinearClsHead):
            raise TypeError('Head of the model should be of type CoSenLinearClsHead')

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper
                   ) -> Dict[str, torch.Tensor]:


        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses[0])  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars, losses[1]
