

_base_ = [
        '../../model/vit.py',
        '../../data/phase1/bepn6.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
]

load_from = None
resume = False
