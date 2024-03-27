#!/bin/bash

cd ..

python train.py ../config/phase2/swin_ros25.py --work-dir ../work_dirs/phase2/swin/ros25/lr_decr/

python train.py ../config/phase2/swin_ros50.py --work-dir ../work_dirs/phase2/swin/ros50/lr_decr/

python train.py ../config/phase2/swin_ros75.py --work-dir ../work_dirs/phase2/swin/ros75/lr_decr/

python train.py ../config/phase2/swin_ros100.py --work-dir ../work_dirs/phase2/swin/ros100/lr_decr/
