#!/bin/bash

cd ..

python train.py ../config/phase4/swin_moco.py --work-dir ../work_dirs/phase4/swin/ssl_moco/lr_decr/

