#!/bin/bash

cd ..

./dist_train.sh ../config/phase_1/vit_sgd0_01.py 3 --work-dir ../work_dirs/phase_1/vit/lr_0.01/

./dist_train.sh ../config/phase_1/vit_sgd0_001.py 3 --work-dir ../work_dirs/phase_1/vit/lr_0.001/

./dist_train.sh ../config/phase_1/vit_sgd_decr.py 3 --work-dir ../work_dirs/phase_1/vit/lr_decr/

