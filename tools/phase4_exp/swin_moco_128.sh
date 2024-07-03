#!/bin/bash

cd ..

python train.py ../config/phase4/swin_moco_128.py --work-dir ../work_dirs/phase4/swin/ssl_moco_128/lr_decr/

