#!/bin/bash

cd ..

python train.py ../config/phase2/swin_ds.py --work-dir ../work_dirs/phase2/swin/ds/lr_decr/

python train.py ../config/phase2/swin_ds.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/ds/0.001/

python train.py ../config/phase2/swin_ds.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/ds/0.01/


