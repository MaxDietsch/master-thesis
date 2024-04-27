_base_ = [
        '../../model/efficientnet_b4_focal.py',
        '../../data/phase2/bepn8.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
