

_base_ = [
        '../../model/swin.py',
        '../../data/phase1/bepn14.py',
        '../../schedule/sgd0_01.py',
        '../../runtime/default.py'
]

load_from = None
resume = False
