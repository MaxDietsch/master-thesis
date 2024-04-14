#!/bin/bash

cd ..

#python train.py ../config/phase2/swin_col.py --work-dir ../work_dirs/phase2/swin/col/lr_decr/

#python train.py ../config/phase2/swin_col.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/col/lr_0.001/

#python train.py ../config/phase2/swin_col.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/col/lr_0.01/

python train.py ../config/phase2/swin_col_100.py --work-dir ../work_dirs/phase2/swin/col_100/lr_decr/

python train.py ../config/phase2/swin_col_100.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/col_100/lr_0.01/

python train.py ../config/phase2/swin_col_100.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/col_100/lr_0.001/


python train.py ../config/phase2/swin_col_45.py --work-dir ../work_dirs/phase2/swin/col_45/lr_decr/

python train.py ../config/phase2/swin_col_45.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/col_45/lr_0.01/

python train.py ../config/phase2/swin_col_45.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/col_45/lr_0.001/


