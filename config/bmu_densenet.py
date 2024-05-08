_base_ = [
        '../model/densenet121.py',
        '../data/bepnwom2.py',
        '../schedule/bmu_sgd.py',
        '../runtime/default.py'
        ]

load_from = None
resume = False
