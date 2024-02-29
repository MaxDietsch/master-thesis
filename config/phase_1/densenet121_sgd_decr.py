_base_ = [
        '../../model/densenet121.py',
        '../../data/phase1/bepn16.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
