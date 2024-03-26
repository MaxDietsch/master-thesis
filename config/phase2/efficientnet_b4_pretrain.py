_base_ = [
        '../../model/efficientnet_b4_pretrain.py',
        '../../data/phase2/pretrain8.py',
        '../../schedule/sgd_pretrain.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
