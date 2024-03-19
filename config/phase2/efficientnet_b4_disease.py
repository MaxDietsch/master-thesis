_base_ = [
        '../../model/efficientnet_b4_disease.py',
        '../../data/phase2/bepn8_disease.py',
        '../../schedule/sgd_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
