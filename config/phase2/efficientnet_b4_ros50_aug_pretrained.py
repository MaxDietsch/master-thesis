_base_ = [
        '../../model/efficientnet_b4_pretrained.py',
        '../../data/phase2/bepn8_ros50.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = '../../work_dirs/phase2/efficiennet_b4/ros50_aug_pretrained/lr_decr/epoch_85.pth'
