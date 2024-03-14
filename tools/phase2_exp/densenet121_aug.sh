#!/bin/bash

cd ..

python train.py ../config/phase2/densenet121_aug.py --work-dir ../work_dirs/phase2/densenet121/aug/sgd_decr

python train.py ../config/phase2/densenet121_aug2.py --work-dir ../work_dirs/phase2/densenet121/aug2/sgd_decr

python train.py ../config/phase2/densenet121_aug3.py --work-dir ../work_dirs/phase2/densenet121/aug3/sgd_decr

python train.py ../config/phase2/densenet121_aug4.py --work-dir ../work_dirs/phase2/densenet121/aug4/sgd_decr

