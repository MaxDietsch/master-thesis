_base_ = [
        '../../model/swin_128.py',
        '../../data/phase4/test_128.py',
        #'../../schedule/sgd_decr.py',
        '../../runtime/default.py'
]

val_cfg = dict()
test_cfg = dict()


load_from = None
resume = False
