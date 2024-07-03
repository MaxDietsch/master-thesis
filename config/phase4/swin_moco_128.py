_base_ = [
        '../../model/swin_mocov2_128.py',
        '../../data/phase4/selfsup8_128.py',
        '../../schedule/sgd_ssl2.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
