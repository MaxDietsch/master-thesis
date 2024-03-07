

_base_ = [
        '../../model/resnet50.py',
        '../../data/phase1/test.py',
        #'../../schedule/sgd0_01.py',
        '../../runtime/default.py'
]

val_cfg = dict()
test_cfg = dict()

load_from = None
resume = False

