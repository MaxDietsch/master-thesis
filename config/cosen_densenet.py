_base_ = [
        '../model/cosen_densenet121.py',
        '../data/bepnwom16.py',
        '../schedule/cosen_sgd.py',
        '../runtime/default.py'
        ]

load_from = None
resume = False
