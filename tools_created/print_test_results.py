import os
import json

def find_and_print_json_values(root_dir):
    # Walk through the directory structure starting at 'root_dir'
    for d1 in os.listdir(root_dir):
        d1_path = os.path.join(root_dir, d1)
        if os.path.isdir(d1_path):  # Ensure d1 is a directory
            for d2 in os.listdir(d1_path):
                d2_path = os.path.join(d1_path, d2)
                if os.path.isdir(d2_path):  # Ensure d2 is a directory
                    for file in os.listdir(d2_path):
                        if file.endswith('.json'):  # Check if the file is a JSON file
                            file_path = os.path.join(d2_path, file)
                            with open(file_path, 'r') as json_file:
                                try:
                                    data = json.load(json_file)
                                    # Assuming 'accuracy' and 'recall' are top-level keys in the JSON file
                                    accuracy = data.get('accuracy', 'N/A')
                                    recall = data.get('recall', 'N/A')
                                    print(f"{d1}: Accuracy: {accuracy}, Recall: {recall}\n")
                                except json.JSONDecodeError:
                                    print(f"Error reading JSON file: {file_path}")
    # Print the separator line
    print(100 * "-")

# Example usage
specified_directory = "../work_dirs/phase_1/resnet50/test"
find_and_print_json_values(specified_directory)

