#!/bin/bash

cd ..

python train.py ../config/phase_1/densenet121_sgd_decr.py --work-dir ../work_dirs/phase_1/densenet121/lr_decr_new/


python train.py ../config/phase_1/efficientnet_b4_sgd_decr.py --work-dir ../work_dirs/phase_1/efficientnet_b4/lr_decr_new/
