_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase2/bepn1_bmu.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
