#!/bin/bash

cd ..

python train.py ../config/phase2/swin_rus1.py --work-dir ../work_dirs/phase2/swin/rus1/lr_decr/

python train.py ../config/phase2/swin_rus3.py --work-dir ../work_dirs/phase2/swin/rus3/lr_decr/

python train.py ../config/phase2/swin_rus5.py --work-dir ../work_dirs/phase2/swin/rus5/lr_decr/

python train.py ../config/phase2/swin_rus10.py --work-dir ../work_dirs/phase2/swin/rus10/lr_decr/


python train.py ../config/phase2/swin_rus1.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/rus1/0.001/

python train.py ../config/phase2/swin_rus3.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/rus3/0.001/

python train.py ../config/phase2/swin_rus5.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/rus5/0.001/

python train.py ../config/phase2/swin_rus10.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/rus10/0.001/

python train.py ../config/phase2/swin_rus1.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/rus1/0.01/

python train.py ../config/phase2/swin_rus3.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/rus3/0.01/

python train.py ../config/phase2/swin_rus5.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/rus5/0.01/

python train.py ../config/phase2/swin_rus10.py --lr 0.01 --work-dir ../work_dirs/phase2/swin/rus10/0.01/
