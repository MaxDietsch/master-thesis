_base_ = [
        '../../model/densenet121.py',
        '../../data/bepn16.py',
        '../../schedule/sgd0_01.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
