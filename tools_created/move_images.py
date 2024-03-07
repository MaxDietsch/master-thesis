import os
import shutil

"""
This script takes a directory (source_dir) and all images with this are copied to 
a new directory (target_directory).
"""


def count_existing_images(directory, file_extensions):
        return sum(1 for filename in os.listdir(directory) if any(filename.endswith(ext) for ext in file_extensions))


def move_images(source_directory, target_directory, file_extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    count = count_existing_images(target_directory, file_extensions)
    dir_name = os.path.basename(target_directory)


    for filename in os.listdir(source_directory):
        if any(filename.endswith(ext) for ext in file_extensions):
            new_filename = f"{dir_name}_{count}{os.path.splitext(filename)[1]}"

            shutil.copy(os.path.join(source_directory, filename),
                        os.path.join(target_directory, new_filename))
            print(f"Moved: {filename} to {new_filename}")
            count += 1


source_dir = '../../ALL_Kvasir+Gastro/barretts'  
target_dir = '../../B_E_P_N-without-mix/barretts'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/barretts-short-segment'  
target_dir = '../../B_E_P_N-without-mix/barretts'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/esophagitis-a'  
target_dir = '../../B_E_P_N-without-mix/esophagitis'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/esophagitis-b-d'  
target_dir = '../../B_E_P_N-without-mix/esophagitis'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/pylorus'  
target_dir = '../../B_E_P_N-without-mix/normal'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/z-line'  
target_dir = '../../B_E_P_N-without-mix/normal'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/retroflex-stomach'  
target_dir = '../../B_E_P_N-without-mix/normal'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/gastrovision/Barrett\'s esophagus'  
target_dir = '../../B_E_P_N-without-mix/barretts'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/gastrovision/Gastric polyps'  
target_dir = '../../B_E_P_N-without-mix/polyps'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/gastrovision/Normal esophagus'  
target_dir = '../../B_E_P_N-without-mix/normal'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/gastrovision/Pylorus'  
target_dir = '../../B_E_P_N-without-mix/normal'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/gastrovision/Esophagitis'  
target_dir = '../../B_E_P_N-without-mix/esophagitis'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/gastrovision/Gastroesophageal_junction_normal z-line'  
target_dir = '../../B_E_P_N-without-mix/normal'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/gastrovision/Normal stomach'  
target_dir = '../../B_E_P_N-without-mix/normal'
move_images(source_dir, target_dir)

source_dir = '../../ALL_Kvasir+Gastro/gastrovision/Duodenal bulb'  
target_dir = '../../B_E_P_N-without-mix/normal'
move_images(source_dir, target_dir)
