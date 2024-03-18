#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_aug.py --work-dir ../work_dirs/phase2/efficientnet_b4/aug/lr_decr/

python train.py ../config/phase2/efficientnet_b4_aug2.py --work-dir ../work_dirs/phase2/efficientnet_b4/aug2/lr_decr/

python train.py ../config/phase2/efficientnet_b4_aug3.py --work-dir ../work_dirs/phase2/efficientnet_b4/aug3/lr_decr/

python train.py ../config/phase2/efficientnet_b4_aug4.py --work-dir ../work_dirs/phase2/efficientnet_b4/aug4/lr_decr/

