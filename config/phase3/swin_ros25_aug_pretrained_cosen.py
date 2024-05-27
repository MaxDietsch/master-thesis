_base_ = [
        '../../model/swin_cosen.py',
        '../../data/phase3/bepn14_ros25_cosen.py',
        '../../schedule/sgd_cosen.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
