_base_ = [
        '../../model/densenet121.py',
        '../../data/phase2/bepn16_aug2.py',
        '../../schedule/sgd0_01.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
