import torch
from mmpretrain import get_model

model = get_model('densenet121_3rdparty_in1k', pretrained=True)


print(model)
