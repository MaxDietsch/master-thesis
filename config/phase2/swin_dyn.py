_base_ = [
        '../../model/swin.py',
        '../../data/phase2/bepn14_dyn.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
