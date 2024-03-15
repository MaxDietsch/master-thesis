_base_ = [
        '../../model/densenet121.py',
        '../../data/phase2/bepn16_aug2.py',
        '../../schedule/sgd_0.01.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
