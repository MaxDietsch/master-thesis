#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_ros25_aug_pretrained.py --work-dir ../work_dirs/phase2/efficientnet_b4/ros25_aug_pretrained/lr_decr/

python train.py ../config/phase2/efficientnet_b4_ros50_aug_pretrained.py --work-dir ../work_dirs/phase2/efficientnet_b4/ros50_aug_pretrained/lr_decr/

python train.py ../config/phase2/efficientnet_b4_ros75_aug_pretrained.py --work-dir ../work_dirs/phase2/efficientnet_b4/ros75_aug_pretrained/lr_decr/

python train.py ../config/phase2/efficientnet_b4_ros100_aug_pretrained.py --work-dir ../work_dirs/phase2/efficientnet_b4/ros100_aug_pretrained/lr_decr/


python train.py ../config/phase2/efficientnet_b4_ros25_aug_pretrained.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/ros25_aug_pretrained/lr_0.01/

python train.py ../config/phase2/efficientnet_b4_ros50_aug_pretrained.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/ros50_aug_pretrained/lr_0.01/

python train.py ../config/phase2/efficientnet_b4_ros75_aug_pretrained.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/ros75_aug_pretrained/lr_0.01/

python train.py ../config/phase2/efficientnet_b4_ros100_aug_pretrained.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/ros100_aug_pretrained/lr_0.01/


python train.py ../config/phase2/efficientnet_b4_ros25_aug_pretrained.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/ros25_aug_pretrained/lr_0.001/

python train.py ../config/phase2/efficientnet_b4_ros50_aug_pretrained.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/ros50_aug_pretrained/lr_0.001/

python train.py ../config/phase2/efficientnet_b4_ros75_aug_pretrained.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/ros75_aug_pretrained/lr_0.001/

python train.py ../config/phase2/efficientnet_b4_ros100_aug_pretrained.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/ros100_aug_pretrained/lr_0.001/


