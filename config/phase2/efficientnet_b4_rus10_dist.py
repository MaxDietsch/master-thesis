_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase2/bepn2_rus10.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
