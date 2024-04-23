#!/bin/bash

cd ..

python train.py ../config/phase2/swin_ros25_aug_pretrained.py --work-dir ../work_dirs/phase2/swin/ros25_aug_pretrained/lr_decr/

python train.py ../config/phase2/swin_ros50_aug_pretrained.py --work-dir ../work_dirs/phase2/swin/ros50_aug_pretrained/lr_decr/

python train.py ../config/phase2/swin_ros75_aug_pretrained.py --work-dir ../work_dirs/phase2/swin/ros75_aug_pretrained/lr_decr/

python train.py ../config/phase2/swin_ros100_aug_pretrained.py --work-dir ../work_dirs/phase2/swin/ros100_aug_pretrained/lr_decr/


python train.py ../config/phase2/swin_ros25_aug_pretrained.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/ros25_aug_pretrained/lr_0.01/

python train.py ../config/phase2/swin_ros50_aug_pretrained.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/ros50_aug_pretrained/lr_0.01/

python train.py ../config/phase2/swin_ros75_aug_pretrained.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/ros75_aug_pretrained/lr_0.01/

python train.py ../config/phase2/swin_ros100_aug_pretrained.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/ros100_aug_pretrained/lr_0.01/


python train.py ../config/phase2/swin_ros25_aug_pretrained.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/ros25_aug_pretrained/lr_0.001/

python train.py ../config/phase2/swin_ros50_aug_pretrained.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/ros50_aug_pretrained/lr_0.001/

python train.py ../config/phase2/swin_ros75_aug_pretrained.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/ros75_aug_pretrained/lr_0.001/

python train.py ../config/phase2/swin_ros100_aug_pretrained.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/ros100_aug_pretrained/lr_0.001/
