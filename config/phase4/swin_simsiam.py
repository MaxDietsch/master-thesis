_base_ = [
        '../../model/swin_simsiam.py',
        '../../data/phase4/selfsup7.py',
        '../../schedule/sgd_ssl.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
