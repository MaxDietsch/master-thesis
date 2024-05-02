_base_ = [
        '../../model/efficientnet_b4_pretrained_focal2.py',
        '../../data/phase2/bepn8_ros75.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
