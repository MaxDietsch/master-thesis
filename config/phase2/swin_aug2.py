

_base_ = [
        '../../model/swin.py',
        '../../data/phase1/bepn14_aug2.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
]

load_from = None
resume = False
