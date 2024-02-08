#from models.densenet import DenseNet121
import torch

from models.densenet import DenseNet121
from mmpretrain.registry import MODELS


model = DenseNet121()
"""
model = dict(
            backbone=dict(arch='121', type='DenseNet'),
            head=dict(
            in_channels=1024,
            loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
            num_classes=4,
            topk=1,
            type='LinearClsHead'),
            neck=dict(type='GlobalAveragePooling'),
            type='ImageClassifier'
        )
model = MODELS.build(model)
"""
#print(model)

weights_path = '../work_dirs/densetnet121_sgd_bepnwom_default/epoch_400.pth'
"""
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print ("#" * 100)


state_dict = torch.load(weights_path)['state_dict']

print("Keys in the state dictionary:")
for key in state_dict.keys():
        print(key)
"""

model.load_state_dict(torch.load(weights_path)['state_dict'])

#import inspect
#from mmpretrain import get_model

#print(inspect.getsource(get_model))


#print(model)


