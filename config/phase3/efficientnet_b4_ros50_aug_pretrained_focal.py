_base_ = [
        '../../model/efficientnet_b4_pretrained_focal.py',
        '../../data/phase2/bepn8_ros50.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
