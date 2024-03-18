_base_ = [
        '../../model/swin.py',
        '../../data/phase2/test.py',
        #'../../schedule/sgd0_001.py',
        '../../runtime/default.py'
        ]

val_cfg = dict()
test_cfg = dict()

load_from = None
resume = False
