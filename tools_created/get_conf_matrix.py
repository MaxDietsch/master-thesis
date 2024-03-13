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

cms = {"lr_0.01": [], 'lr_0.001': [], 'lr_decr': []}

def find_json_values(root_dir):
    # Walk through the directory structure starting at 'root_dir'
    for d1 in os.listdir(root_dir):
        d1_path = os.path.join(root_dir, d1)
        if os.path.isdir(d1_path):  # Ensure d1 is a directory
            for d2 in os.listdir(d1_path):
                d2_path = os.path.join(d1_path, d2)
                for file in os.listdir(d2_path):

                    if file.endswith('.pt'):
                        file_path = os.path.join(d2_path, file)
                        loaded_cm = torch.load(file_path)
                        cms[d1].append(loaded_cm)
    # Print the separator line
    print('reading finished')

def calculate_average():

    for key in cms: 
        cm_mean = torch.mean(torch.stack(cms[key]), dim = 0)
        cm_std = torch.std(torch.stack(cms[key]), dim = 0)
        
        
        with open(txt_path, 'a') as file:
            file.write(f"\n\n(Average Confusion Matrix of) Model: {model} with schedule: {key} \n")
            
            for cm in zip(cm_mean, cm_std):
                cm_mean = np.round(cm_mean.cpu().numpy(), 4)
                
                torch.save(cm_mean, specified_directory + '/' + key + '_cm.pt')

                #file.write(f"{metric} \n mean: \t {cm_mean} \n std: \t {cm_std} \n\n")
            

# Example usage
specified_directory = "../work_dirs/phase1/resnet50/test"
txt_path = '../work_dirs/phase1/results.txt'
model = 'ResNet50'
find_json_values(specified_directory)
calculate_average()

print('The results are written to the specified file!')

