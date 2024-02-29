_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase1/bepn8.py',
        '../../schedule/sgd0_001.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
