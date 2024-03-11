

_base_ = [
        '../model/dos_densenet121.py',
        '../data/phase2/bepn8_dos.py',
        '../schedule/dos_sgd.py',
        '../runtime/default.py'
]

#load_from = '../work_dirs/densetnet121_sgd_bepnwom_default/epoch_400.pth'
load_from = None
resume = False
