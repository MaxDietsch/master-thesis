_base_ = [
        '../../model/efficientnet_b4_cosen.py',
        '../../data/phase3/bepn8_ros50_cosen.py',
        '../../schedule/sgd_cosen.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
