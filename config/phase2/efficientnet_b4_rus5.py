_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase2/bepn8_rus5.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
