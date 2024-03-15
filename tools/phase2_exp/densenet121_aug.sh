#!/bin/bash

cd ..

python train.py ../config/phase2/densenet121_aug.py --work-dir ../work_dirs/phase2/densenet121/aug/lr_0.01/

python train.py ../config/phase2/densenet121_aug2.py --work-dir ../work_dirs/phase2/densenet121/aug2/lr_0.01/

python train.py ../config/phase2/densenet121_aug3.py --work-dir ../work_dirs/phase2/densenet121/aug3/lr_0.01/

python train.py ../config/phase2/densenet121_aug4.py --work-dir ../work_dirs/phase2/densenet121/aug4/lr_0.01/

