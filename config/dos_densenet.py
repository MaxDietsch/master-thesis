

_base_ = [
        '../model/dos_densenet121.py',
        '../data/test.py',
        '../schedule/sgd.py',
        '../runtime/default.py'
]

load_from = '../work_dirs/densetnet121_sgd_bepnwom_default/epoch_400.pth'
resume = False
