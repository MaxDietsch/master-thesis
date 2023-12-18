
_base_ = [
        '../model/resnet50.py',
        '../data/test.py',
        '../schedule/sgd.py',
        '../runtime/default.py'
]

load_from = None
resume = False
