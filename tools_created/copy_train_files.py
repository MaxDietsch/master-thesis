
import os 
import shutil

"""
This script takes a directory (ROOT) and searches for meta/train.txt files. 
Train.txt should look like this: 
../../B_E_P_N/train/esophagitis_603.jpg 3, where B_E_P_N is the ROOT.
All images specified in this file should be present in ROOT/class_name.
The images are copied to a new directory: dataset_name/class_name.
"""

ROOT = '../../B_E_P_N/'

# create new dataset directory
dataset_name = 'B_E_P_N_aug2'
new_data_dir = os.path.join(ROOT + '../', dataset_name)
os.makedirs(new_data_dir, exist_ok = True)

# get the file names
train_file_path = os.path.join(ROOT, 'meta/train.txt')
extracted_files = {'esophagitis': [], 'polyps': [], 'barretts': [], 'normal': []}

with open(train_file_path, 'r') as file:
    for line in file:
        filename = line.split('/')[-1].split(' ')[0]

        class_name = filename.split('_')[0]

        extracted_files[class_name].append(filename)


# copy files to directory 
for key, filenames in extracted_files.items():
    os.makedirs(os.path.join(new_data_dir, key), exist_ok = True)
    for filename in filenames:
        source_path = os.path.join(ROOT, key, filename)
        destination_path = os.path.join(new_data_dir, key, filename)
        shutil.copy2(source_path, destination_path)

print('Copied all train files to new dataset directory. \n Now they can be augmented')









