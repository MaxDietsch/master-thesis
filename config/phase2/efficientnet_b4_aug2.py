_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase2/bepn8_aug2.py',
        '../../schedule/sgd0_001.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
