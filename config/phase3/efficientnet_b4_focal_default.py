_base_ = [
        '../../model/efficientnet_b4_focal.py',
        '../../data/phase2/bepn8_aug.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
