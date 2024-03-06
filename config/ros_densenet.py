_base_ = [
        '../model/densenet121.py',
        '../data/ros_bepn16.py',
        '../schedule/sgd.py',
        '../runtime/default.py'
        ]

load_from = None
resume = False
