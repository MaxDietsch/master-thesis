#!/bin/bash

cd ..

python train.py ../config/phase2/swin_aug.py --work-dir ../work_dirs/phase2/swin/aug/lr_decr/

python train.py ../config/phase2/swin_aug2.py --work-dir ../work_dirs/phase2/swin/aug2/lr_decr/

python train.py ../config/phase2/swin_aug3.py --work-dir ../work_dirs/phase2/swin/aug3/lr_decr/

python train.py ../config/phase2/swin_aug4.py --work-dir ../work_dirs/phase2/swin/aug4/lr_decr/

