

_base_ = [
        '../../model/vit.py',
        '../../data/phase1/test.py',
        #'../../schedule/sgd_decr.py',
        '../../runtime/default.py'
]

val_cfg = dict()
test_cfg = dict()


load_from = None
resume = False
