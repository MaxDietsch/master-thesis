#!/bin/bash

cd ..

python train.py ../config/phase4/efficientnet_b4_relative_loc.py --work-dir ../work_dirs/phase4/efficientnet_b4/ssl_relative_loc/lr_decr/

