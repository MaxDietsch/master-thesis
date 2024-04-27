#!/bin/bash

cd ..

python train.py ../config/phase3/efficientnet_b4_focal_default.py --work-dir ../work_dirs/phase3/efficientnet_b4/focal_default/lr_decr/

python train.py ../config/phase3/efficientnet_b4_focal_default.py --lr 0.01 --work-dir ../work_dirs/phase3/efficientnet_b4/focal_default/lr_0.01/

python train.py ../config/phase3/efficientnet_b4_focal_default.py --lr 0.001 --work-dir ../work_dirs/phase3/efficientnet_b4/focal_default/lr_0.001/


