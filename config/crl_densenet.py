_base_ = [
        '../model/cosen_densenet121.py',
        '../data/bepnwom16.py',
        '../schedule/crl_sgd.py',
        '../runtime/default.py'
        ]

load_from = None
resume = False
