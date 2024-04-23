_base_ = [
        '../../model/swin_pretrained.py',
        '../../data/phase2/bepn14_ros25.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
