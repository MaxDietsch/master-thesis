_base_ = [
        '../../model/efficientnet_b7.py',
        '../../data/phase1/bepn6.py',
        '../../schedule/sgd0_01.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
