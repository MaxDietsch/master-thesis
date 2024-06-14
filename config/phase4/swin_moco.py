_base_ = [
        '../../model/swin_mocov2.py',
        '../../data/phase4/selfsup14.py',
        '../../schedule/sgd_ssl2.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
