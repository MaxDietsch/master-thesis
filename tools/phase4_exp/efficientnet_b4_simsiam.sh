#!/bin/bash

cd ..

python train.py ../config/phase4/efficientnet_b4_simsiam.py --work-dir ../work_dirs/phase4/efficientnet_b4/ssl_simsiam/lr_decr/

