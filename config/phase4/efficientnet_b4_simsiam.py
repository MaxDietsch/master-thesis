_base_ = [
        '../../model/efficientnet_b4_simsiam.py',
        '../../data/phase4/selfsup8.py',
        '../../schedule/sgd_ssl.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
