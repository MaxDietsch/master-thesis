#!/bin/bash

cd ..

python train.py ../config/phase2/efficientnet_b4_aug.py --work-dir ../work_dirs/phase2/efficientnet_b4/aug/

python train.py ../config/phase2/efficientnet_b4_aug2.py --work-dir ../work_dirs/phase2/efficientnet_b4/aug2/

python train.py ../config/phase2/efficientnet_b4_aug3.py --work-dir ../work_dirs/phase2/efficientnet_b4/aug3/

python train.py ../config/phase2/efficientnet_b4_aug4.py --work-dir ../work_dirs/phase2/efficientnet_b4/aug4/

