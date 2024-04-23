#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_ds.py --work-dir ../work_dirs/phase2/efficientnet_b4/ds/lr_decr/

python train.py ../config/phase2/efficientnet_b4_ds.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/ds/lr_0.01/

python train.py ../config/phase2/efficientnet_b4_ds.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/ds/lr_0.001/


