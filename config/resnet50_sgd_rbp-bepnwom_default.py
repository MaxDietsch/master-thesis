_base_ = [
        '../model/resnet50.py',
        '../data/rpb_bepnwom32.py',
        '../schedule/sgd.py',
        '../runtime/default.py'
]

load_from = None
resume = False

