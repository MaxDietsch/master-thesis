_base_ = [
        '../../model/swin_focal.py',
        '../../data/phase1/bepn14.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
