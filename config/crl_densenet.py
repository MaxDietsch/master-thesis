_base_ = [
        '../model/crl_densenet121.py',
        '../data/bepnwom16.py',
        '../schedule/sgd.py',
        '../runtime/default.py'
        ]

load_from = None
resume = False
