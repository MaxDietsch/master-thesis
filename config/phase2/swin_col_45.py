_base_ = [
        '../../model/swin.py',
        '../../data/phase2/bepn14_col_45.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = None
