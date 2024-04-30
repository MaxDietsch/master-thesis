_base_ = [
        '../../model/swin_focal.py',
        '../../data/phase2/bepn14_aug.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
