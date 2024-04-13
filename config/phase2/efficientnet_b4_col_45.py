_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase2/bepn8_col_45.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = '../work_dirs/phase2/efficientnet_b4/col_45/lr_0.001'
