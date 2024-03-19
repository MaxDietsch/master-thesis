#!/bin/bash

cd ..

python train.py ../config/phase2/swin_healthy.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/healthy/0.001/

python train.py ../config/phase2/swin_healthy.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/healthy/0.01/

python train.py ../config/phase2/swin_healthy.py --lr decr --work-dir ../work_dirs/phase2/swin/healthy/decr/

python train.py ../config/phase2/swin_disease.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/disease/0.001/

python train.py ../config/phase2/swin_disease.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/disease/0.01/

python train.py ../config/phase2/swin_disease.py --lr decr --work-dir ../work_dirs/phase2/swin/disease/decr/

