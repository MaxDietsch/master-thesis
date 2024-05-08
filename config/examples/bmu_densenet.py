_base_ = [
        '../model/bmu_densenet121.py',
        '../data/bepnwom2.py',
        '../schedule/bmu_sgd.py',
        '../runtime/default.py'
        ]

load_from = None
resume = False
