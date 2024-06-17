#!/bin/bash

cd ..

python train.py ../config/phase4/swin_pretrained_simsiam.py --work-dir ../work_dirs/phase4/swin/pretrained_simsiam/lr_decr/

python train.py ../config/phase4/swin_pretrained_simsiam.py --lr 0.01 --work-dir ../work_dirs/phase4/swin/pretrained_simsiam/lr_0.01/

python train.py ../config/phase4/swin_pretrained_simsiam.py --lr 0.001 --work-dir ../work_dirs/phase4/swin/pretrained_simsiam/lr_0.001/


