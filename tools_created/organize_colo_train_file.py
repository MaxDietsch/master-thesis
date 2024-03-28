import os

# Directory containing the files to read
source_directory = '../../B_E_P_N_COL/col_transformed'

# The file to which the names of all files will be appended
output_file_path = '../../B_E_P_N_COL/meta/train.txt'

# Open the output file in append mode
with open(output_file_path, 'a') as output_file:
    # List every file in the directory
    for file_name in os.listdir(source_directory):
        # Check if it's a file and not a directory
        if os.path.isfile(os.path.join(source_directory, file_name)):
            # Append the file name to the output file
            string = '../../B_E_P_N/train/' + file_name + ' 1'
            output_file.write(string + '\n')

