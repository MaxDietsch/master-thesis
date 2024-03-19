

"""
This script takes a input file (input_file_path, train.txt) with lines like: 
../../B_E_P_N/train/esophagitis_603.jpg 3   -> 3 is the label
This script creates a new file (output_file_path) where all labels greater than 0 are 
clipped to 1 (so that you have a binary classification).
"""


def manipulate_and_save_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    # Manipulate the number after the whitespace in each line
    modified_lines = []
    for line in lines:
        parts = line.rsplit(' ', 1)  # Split each line into two parts
        number = int(parts[1].strip())
        if number > 0: 
            number = 1
            modified_lines.append(f"{parts[0]} {number}\n")
        else:
            modified_lines.append(line)  # Keep the line unchanged if it doesn't match the expected format
    
    # Write the modified content to the output file
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(modified_lines)


#input_file_path = '../../../B_E_P_N/meta/train.txt'
#output_file_path = '../../../B_E_P_N/meta_2phase/train_healthy.txt'  

input_file_path = '../../B_E_P_N/meta/val.txt'
output_file_path = '../../../B_E_P_N/meta_2phase/val_healthy.txt'
manipulate_and_save_file(input_file_path, output_file_path)

input_file_path = '../../B_E_P_N/meta/test.txt'
output_file_path = '../../B_E_P_N/meta_2phase/test_healthy.txt'
manipulate_and_save_file(input_file_path, output_file_path)


