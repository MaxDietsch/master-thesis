_base_ = [
        '../../model/resnet50.py',
        '../../data/phase1/test.py',
        #'../../schedule/sgd0_01.py',
        '../../runtime/default.py'
]

load_from = None
resume = False

