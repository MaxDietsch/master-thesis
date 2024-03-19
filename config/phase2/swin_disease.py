

_base_ = [
        '../../model/swin_disease.py',
        '../../data/phase2/bepn14_disease.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
]

load_from = None
resume = False
