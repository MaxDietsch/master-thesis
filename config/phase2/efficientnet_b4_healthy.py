_base_ = [
        '../../model/efficientnet_b4_healthy.py',
        '../../data/phase2/bepn8_healthy.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
