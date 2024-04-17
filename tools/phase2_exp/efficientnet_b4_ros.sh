#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_ros25.py --work-dir ../work_dirs/phase2/efficientnet_b4/ros25_aug3/lr_decr/

python train.py ../config/phase2/efficientnet_b4_ros50.py --work-dir ../work_dirs/phase2/efficientnet_b4/ros50_aug3/lr_decr/

python train.py ../config/phase2/efficientnet_b4_ros75.py --work-dir ../work_dirs/phase2/efficientnet_b4/ros75_aug3/lr_decr/

python train.py ../config/phase2/efficientnet_b4_ros100.py --work-dir ../work_dirs/phase2/efficientnet_b4/ros100_aug3/lr_decr/


python train.py ../config/phase2/efficientnet_b4_ros25.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/ros25_aug3/lr_0.01/

python train.py ../config/phase2/efficientnet_b4_ros50.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/ros50_aug3/lr_0.01/

python train.py ../config/phase2/efficientnet_b4_ros75.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/ros75_aug3/lr_0.01/

python train.py ../config/phase2/efficientnet_b4_ros100.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/ros100_aug3/lr_0.01/


python train.py ../config/phase2/efficientnet_b4_ros25.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/ros25_aug3/lr_0.001/

python train.py ../config/phase2/efficientnet_b4_ros50.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/ros50_aug3/lr_0.001/

python train.py ../config/phase2/efficientnet_b4_ros75.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/ros75_aug3/lr_0.001/

python train.py ../config/phase2/efficientnet_b4_ros100.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/ros100_aug3/lr_0.001/


