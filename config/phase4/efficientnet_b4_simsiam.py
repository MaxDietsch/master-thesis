_base_ = [
        '../../model/efficientnet_b4_simsiam.py',
        '../../data/phase4/selfsup4.py',
        '../../schedule/sgd_ssl.py',
        '../../runtime/default.py'
        ]

load_from = '../../work_dirs/phase4/efficientnet_b4/ssl_simsiam/lr_decr/epoch_119.pth'
resume = True
