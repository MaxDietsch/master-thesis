import os
from PIL import Image

image_directory = '../../rBP_B_E_P_N-without-mix/barretts' 
rotation_degrees = [90, 180, 270] 

# take image and rotate it about every value in degrees parameter
def rotate_and_save(image_path, degrees):
    with Image.open(image_path) as img:
        for degree in degrees:
            rotated_img = img.rotate(degree, expand=True)
            name, ext = os.path.splitext(image_path)
            new_image_path = f"{name}_{degree}{ext}"
            rotated_img.save(new_image_path)
            print(f"Saved rotated image: {new_image_path}")

def process_images(directory, degrees):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')): 
            file_path = os.path.join(directory, filename)
            rotate_and_save(file_path, degrees)

process_images(image_directory, rotation_degrees)

