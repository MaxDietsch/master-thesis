#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_rus1.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/rus1/0.001/

python train.py ../config/phase2/efficientnet_b4_rus3.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/rus3/0.001/

python train.py ../config/phase2/efficientnet_b4_rus5.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/rus5/0.001/

python train.py ../config/phase2/efficientnet_b4_rus10.py --lr 0.001 --work-dir ../work_dirs/phase2/efficientnet_b4/rus10/0.001/

python train.py ../config/phase2/efficientnet_b4_rus1.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/rus1/0.01/

python train.py ../config/phase2/efficientnet_b4_rus3.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/rus3/0.01/

python train.py ../config/phase2/efficientnet_b4_rus5.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/rus5/0.01/

python train.py ../config/phase2/efficientnet_b4_rus10.py --lr 0.01 --work-dir ../work_dirs/phase2/efficientnet_b4/rus10/0.01/

python train.py ../config/phase2/efficientnet_b4_rus1.py --work-dir ../work_dirs/phase2/efficientnet_b4/rus1/lr_decr/

python train.py ../config/phase2/efficientnet_b4_rus3.py --work-dir ../work_dirs/phase2/efficientnet_b4/rus3/lr_decr/

python train.py ../config/phase2/efficientnet_b4_rus5.py --work-dir ../work_dirs/phase2/efficientnet_b4/rus5/lr_decr/

python train.py ../config/phase2/efficientnet_b4_rus10.py --work-dir ../work_dirs/phase2/efficientnet_b4/rus10/lr_decr/


