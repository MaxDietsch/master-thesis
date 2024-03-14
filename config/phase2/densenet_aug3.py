_base_ = [
        '../../model/densenet121.py',
        '../../data/phase2/bepn16_aug3.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
