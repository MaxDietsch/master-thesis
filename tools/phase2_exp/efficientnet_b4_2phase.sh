#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_healthy.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/healthy/0.001/

python train.py ../config/phase2/efficientnet_b4_healthy.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/healthy/0.01/

python train.py ../config/phase2/efficientnet_b4_healthy.py --lr decr --work-dir ../work_dirs/phase2/efficientnet_b4/healthy/decr/

python train.py ../config/phase2/efficientnet_b4_disease.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/disease/0.001/

python train.py ../config/phase2/efficientnet_b4_disease.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/disease/0.01/

python train.py ../config/phase2/efficientnet_b4_disease.py --lr decr --work-dir ../work_dirs/phase2/efficientnet_b4/disease/decr/

