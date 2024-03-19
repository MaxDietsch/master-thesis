

_base_ = [
        '../../model/swin_healthy.py',
        '../../data/phase2/pretrain14.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
]

load_from = None
resume = False
