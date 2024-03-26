

_base_ = [
        '../../model/swin_pretrain.py',
        '../../data/phase2/pretrain14.py',
        '../../schedule/sgd_pretrain.py',
        '../../runtime/default.py'
]

load_from = None
resume = False
