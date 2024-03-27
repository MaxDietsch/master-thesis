_base_ = [
        '../../model/swin.py',
        '../../data/phase2/bepn14_ros100.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = '../work_dirs/phase2/swin/ros100/lr_0.001/epoch_77.pth'
resume = True
