#!/bin/bash

cd ..

python train.py ../config/phase2/swin_col.py --work-dir ../work_dirs/phase2/swin/col/lr_decr/

python train.py ../config/phase2/swin_col.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/col/0.001/

python train.py ../config/phase2/swin_col.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/col/0.01/


