_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase2/pretrain32.py',
        '../../schedule/sgd_pretrain.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
