#!/bin/bash

cd ..

python train.py ../config/phase2/swin_aug.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/aug/0.001/

python train.py ../config/phase2/swin_aug2.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/aug2/0.001/

python train.py ../config/phase2/swin_aug3.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/aug3/0.001/

python train.py ../config/phase2/swin_aug4.py --lr 0.001 --work-dir ../work_dirs/phase2/swin/aug4/0.001/

