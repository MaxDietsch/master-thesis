

_base_ = [
        '../../model/swin.py',
        '../../data/phase1/bepn6.py',
        '../../schedule/sgd0_001.py',
        '../../runtime/default.py'
]

load_from = None
resume = False
