#!/bin/bash

cd ..

python train.py ../config/phase4/swin_simsiam.py --work-dir ../work_dirs/phase4/swin/ssl_simsiam/lr_decr/

