_base_ = [
        '../../model/swin_pretrained_focal2.py',
        '../../data/phase2/bepn14_ros25.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
