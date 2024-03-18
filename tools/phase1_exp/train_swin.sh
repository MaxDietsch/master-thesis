#!/bin/bash

cd ..

python train.py ../config/phase1/swin_sgd0_01.py --work-dir ../work_dirs/phase1/swin/lr_0.01/

python train.py ../config/phase1/swin_sgd0_001.py --work-dir ../work_dirs/phase1/swin/lr_0.001/

python train.py ../config/phase1/swin_sgd_decr.py --work-dir ../work_dirs/phase1/swin/lr_decr/
