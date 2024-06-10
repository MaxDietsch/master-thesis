#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_pretrained_moco.py --work-dir ../work_dirs/phase4/efficientnet_b4/pretrained_moco/lr_decr/

python train.py ../config/phase2/efficientnet_b4_pretrained_moco.py --lr 0.01 --work-dir ../work_dirs/phase4/efficientnet_b4/pretrained_moco/lr_0.01/

python train.py ../config/phase2/efficientnet_b4_pretrained_moco.py --lr 0.001 --work-dir ../work_dirs/phase4/efficientnet_b4/pretrained_moco/lr_0.001/


