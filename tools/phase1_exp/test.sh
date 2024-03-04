#!/bin/bash

cd ..

model="resnet50"
lr_array=("lr_0.01" "lr_0.001" "lr_decr")
epoch_lr0_01="60 98"
epoch_lr0_001="68 100"
epoch_lr_decr="60 62 94"

epochs=("$epoch_lr0_01" "$epoch_lr0_001" "$epoch_lr_decr")

# iterate over outer array
for i in "${!epochs[@]}"
do 
    IFS=' ' read -r -a innerArray <<< "${epochs[$i]}"
    # Iterate over inner array of integers
    for j in "${!innerArray[@]}"
    do
        epoch="${innerArray[$j]}"
        
        python test.py ../config/phase_1/${model}_sgd_decr.py ../work_dirs/phase_1/${model}/${lr_array[$i]}/epoch_"$epoch".pth --out ../work_dirs/phase_1/${model}/test/${lr_array[$i]}_epoch_"$epoch" --out-item metrics

