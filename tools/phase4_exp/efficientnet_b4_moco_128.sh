#!/bin/bash

cd ..

python train.py ../config/phase4/efficientnet_b4_moco_128.py --work-dir ../work_dirs/phase4/efficientnet_b4/ssl_moco_128/lr_decr/

