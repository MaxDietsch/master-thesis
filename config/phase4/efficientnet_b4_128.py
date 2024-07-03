_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase4/bepn8_128.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
