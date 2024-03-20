_base_ = [
        '../../model/efficientnet_b4.py',
        '../../data/phase2/bepn1.py',
        '../../schedule/bmu_decr.py',
        '../../runtime/default.py'
        ]

load_from = None
resume = False
