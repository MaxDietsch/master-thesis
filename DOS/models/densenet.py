
import torch.nn as nn

from models.backbones.densenet import DenseNet
from models.necks.gap import GlobalAveragePooling
from models.heads.linearclshead import LinearClsHead

class DenseNet121(nn.Module):

    def __init__(self):
        super(DenseNet121, self).__init__()

        self.backbone = DenseNet()
        self.neck = GlobalAveragePooling()
        self.head = LinearClsHead(1024, 4)

    def forward(self, x):
        x = self.head(self.neck(self.backbone(x)))
        return x

