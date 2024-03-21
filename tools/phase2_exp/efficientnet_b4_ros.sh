#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_ros25.py --lr decr --work-dir ../work_dirs/phase2/efficientnet_b4/ros25/decr/

python train.py ../config/phase2/efficientnet_b4_ros50.py --lr decr --work-dir ../work_dirs/phase2/efficientnet_b4/ros50/decr/

python train.py ../config/phase2/efficientnet_b4_ros75.py --lr decr --work-dir ../work_dirs/phase2/efficientnet_b4/ros75/decr/

python train.py ../config/phase2/efficientnet_b4_ros100.py --lr decr --work-dir ../work_dirs/phase2/efficientnet_b4/ros100/decr/


