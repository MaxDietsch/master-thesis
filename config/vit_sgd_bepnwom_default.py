

_base_ = [
        '../model/vit.py',
        '../data/bepnwom16.py',
        '../schedule/sgd.py',
        '../runtime/default.py'
]

load_from = None
resume = False
