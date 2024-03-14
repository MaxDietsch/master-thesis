_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase2/bepn16_aug3.py',
        '../../schedule/sgd0_01.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
