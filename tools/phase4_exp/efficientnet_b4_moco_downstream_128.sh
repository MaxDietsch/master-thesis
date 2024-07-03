#!/bin/bash

cd ..

python train.py ../config/phase4/efficientnet_b4_pretrained_moco_128.py --work-dir ../work_dirs/phase4/efficientnet_b4/pretrained_moco_128/lr_decr/

python train.py ../config/phase4/efficientnet_b4_pretrained_moco_128.py --lr 0.01 --work-dir ../work_dirs/phase4/efficientnet_b4/pretrained_moco_128/lr_0.01/

python train.py ../config/phase4/efficientnet_b4_pretrained_moco_128.py --lr 0.001 --work-dir ../work_dirs/phase4/efficientnet_b4/pretrained_moco_128/lr_0.001/


