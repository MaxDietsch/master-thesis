_base_ = [
        '../../model/efficientnet_b4_pretrained_relative_loc.py',
        '../../data/phase1/bepn8.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
