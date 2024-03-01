#!/bin/bash

cd ..

./dist_train.sh ../config/phase_1/swin_sgd0_01.py 3 --work-dir ../work_dirs/phase_1/swin/lr_0.01/

./dist_train.sh ../config/phase_1/swin_sgd0_001.py 3 --work-dir ../work_dirs/phase_1/swin/lr_0.001/

./dist_train.sh ../config/phase_1/swin_sgd_decr.py 3 --work-dir ../work_dirs/phase_1/swin/lr_decr/
