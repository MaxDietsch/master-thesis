

_base_ = [
        '../../model/dos_densenet121.py',
        '../../data/phase2/bepn8_dos.py',
        '../../schedule/dos_sgd.py',
        '../../runtime/default.py'
]

load_from = None
resume = False
