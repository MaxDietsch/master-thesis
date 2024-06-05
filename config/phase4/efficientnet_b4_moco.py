_base_ = [
        '../../model/efficientnet_b4_mocov2.py',
        '../../data/phase4/selfsup8.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
