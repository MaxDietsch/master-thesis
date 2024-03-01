#!/bin/bash

python train.py ../config/phase_1/efficientnet_b4_sgd0_01.py --work-dir ../work_dirs/phase_1/efficientnet_b4/lr_0.01/

python train.py ../config/phase_1/efficientnet_b4_sgd0_001.py --work-dir ../work_dirs/phase_1/efficientnet_b4/lr_0.001/

python train.py ../config/phase_1/efficientnet_b4_sgd_decr.py --work-dir ../work_dirs/phase_1/efficientnet_b4/lr_decr/
