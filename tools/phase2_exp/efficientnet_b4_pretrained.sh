#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_pretrained.py --work-dir ../work_dirs/phase2/efficientnet_b4/pretrained/lr_decr/

python train.py ../config/phase2/efficientnet_b4_pretrained.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/pretrained/lr_0.01/

python train.py ../config/phase2/efficientnet_b4_pretrained.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/pretrained/lr_0.001/


