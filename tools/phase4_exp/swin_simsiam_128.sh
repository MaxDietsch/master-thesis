#!/bin/bash

cd ..

python train.py ../config/phase4/swin_simsiam_128.py --work-dir ../work_dirs/phase4/swin/ssl_simsiam_128/lr_decr/

