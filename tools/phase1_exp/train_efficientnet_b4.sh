#!/bin/bash

cd ..

python train.py ../config/phase1/efficientnet_b4_sgd0_01.py --work-dir ../work_dirs/phase1/efficientnet_b4/lr_0.01/

python train.py ../config/phase1/efficientnet_b4_sgd0_001.py --work-dir ../work_dirs/phase1/efficientnet_b4/lr_0.001/

python train.py ../config/phase1/efficientnet_b4_sgd_decr.py --work-dir ../work_dirs/phase1/efficientnet_b4/lr_decr/
