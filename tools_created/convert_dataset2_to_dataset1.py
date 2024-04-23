import os
import cv2
import numpy as np

"""
That script takes a directory (image_directory) and the mean and standard 
deviations (for each channel) of the images in the specified directory.
It transforms the images, so that the mean and standaard deviations are as
wanted (and inputed in the file, see mean_1, std_1).
The new images are stored in a new directory (output_directory).
"""


# Directory containing the images to transform
image_directory = '../../B_E_P_N_COL/col'
# Output directory for transformed images
output_directory = '../../B_E_P_N_COL/col_transformed'

# Make sure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Dataset 1 stats (target stats)
mean_1 = np.array([151.14, 102.69, 97.74])  #(R, G, B)
std_1 = np.array([70.03, 55.91, 54.73])   # (R, G, B)

# Dataset 2 stats (source stats)
mean_2 = np.array([113.23, 69.13, 48.10])       #(R, G, B)
std_2 = np.array([89.96, 61.53, 50.38])     # (R, G, B) 

for filename in os.listdir(image_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(image_directory, filename)
        img_bgr = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)        

        # Normalize the image to have the mean and std dev of the first dataset
        transformed_img_np = ((img_rgb - mean_2) / std_2) * std_1 + mean_1
        transformed_img_np = np.clip(transformed_img_np, 0, 255)   # Clip to valid range and denormalize

        transformed_img_bgr = cv2.cvtColor(transformed_img_np.astype('uint8'), cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving

        
        # Save the transformed image
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, transformed_img_bgr)

print("Transformation complete. Transformed images saved to:", output_directory)

