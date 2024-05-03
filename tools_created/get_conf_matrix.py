import os
import json
import torch
import numpy as np

"""
This script takes a directory (specified_directory).
It goes one directories deeper from the base directory and searches for .pt files. 
This procedure is done for every subdirectory on the base directory. 
The script will read the .pt files and read in the pytorch tensors in those files.
The average confusion matrix over all found values will be calculated.
The calculated scores will be printed to the screen 
"""
cms = {"lr_decr": []}
#cms = {"lr_0.01": [], 'lr_0.001': [], 'lr_decr': []}

def find_json_values(root_dir):
    # Walk through the directory structure starting at 'root_dir'
    for d1 in os.listdir(root_dir):
        d1_path = os.path.join(root_dir, d1)
        if os.path.isdir(d1_path):  # Ensure d1 is a directory
            for d2 in os.listdir(d1_path):
                d2_path = os.path.join(d1_path, d2)
                for file in os.listdir(d2_path):

                    if file.endswith('.pt') and not file.startswith('avg'):
                        file_path = os.path.join(d2_path, file)
                        loaded_cm = torch.load(file_path).float()
                        cms[d1].append(loaded_cm)
    # Print the separator line
    print('reading finished')

def calculate_average():

    for key in cms:
        cm_mean = torch.mean(torch.stack(cms[key]), dim = 0)
        cm_std = torch.std(torch.stack(cms[key]), dim = 0)
        
        torch.save(cm_mean, specified_directory + '/' + key +  '/cm/' + 'avg_cm.pt')

            

# Example usage
specified_directory = "../work_dirs/phase2/swin/test/ros25_aug_pretrained"
find_json_values(specified_directory)
calculate_average()

print('The results are written to the specified file!')

