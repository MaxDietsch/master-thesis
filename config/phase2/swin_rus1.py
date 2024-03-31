_base_ = [
        '../../model/swin.py',
        '../../data/phase2/bepn14_rus1.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
