import albumentations as A
import cv2
import os 
from matplotlib import pyplot as plt

# Define the augmentations: 

# when rotating the dimension is kept -> distortion (maybe do it different way)
rotate90 = A.SafeRotate(limit = (90, 90), p = 1.0, border_mode = cv2.BORDER_CONSTANT, value = 0)
rotate180 = A.SafeRotate(limit = (180, 180), p = 1.0)
rotate270 = A.SafeRotate(limit = (270, 270), p = 1.0, border_mode = cv2.BORDER_CONSTANT, value = 0)
rotations = [rotate90, rotate180, rotate270]

brightness1 = A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0.2, 0.2), p=1.0)
brightness2 = A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=(-0.2, -0.2), p=1.0)
brightness = [brightness1, brightness2]


elastic1 = A.ElasticTransform(alpha=1, sigma=50, alpha_affine=80, border_mode = cv2.BORDER_CONSTANT, value = 0, p=1.0)
apply_elastic = [0, 2]

# maybe set fit_output to true, is quite similar to elastic1
perspective1 = A.Perspective(scale=(0.2, 0.2), fit_output = False, p=1.0)
apply_perspective = [1, 3]

#crop1 = A.RandomResizedCrop(height=1000, width=1000, scale=(0.8, 1.0), ratio = (0.8, 1.2), interpolation = cv2.INTER_CUBIC, p=1.0)  

# num_images if above trafos are applied: 
# x * 4 (rot) + x * 2 + x * 2 (on 2 rots) + x * 2 (on 2 rots) = 8 * x
def apply_transformations(folder_path):

    # Iterate over all files in the directory
    i = 0
    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load the image
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_transformations = [image]
            
            # apply rotations for each image
            for rot in rotations: 
                transformed = rot(image = image)
                image_transformations.append(transformed['image'])

            for bright in brightness:
                transformed = bright(image = image)
                image_transformations.append(transformed['image'])

            for idx in apply_elastic: 
                transformed = elastic1(image = image_transformations[idx])
                image_transformations.append(transformed['image'])

            for idx in apply_perspective: 
                transformed = perspective1(image = image_transformations[idx])
                image_transformations.append(transformed['image'])

            
            # store the images
            base_name, extension = os.path.splitext(file_path)
            for i, img in enumerate(image_transformations):
                if i == 0:
                    continue
                cv2.imwrite(base_name + f'_{i}' + extension, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            
            """
            # Display the original and transformed images
            plt.figure(figsize=(12, 6))

            plt.subplot(121)
            plt.title('Original')
            plt.imshow(image)
            plt.axis('off')

            plt.subplot(122)
            plt.title('Transformed')
            plt.imshow(transformed_image)
            plt.axis('off')

            plt.show()
            i+=1
            if i >= 5: 
                break
            """

# Example usage
folder_path = '../../../GastroDataset/B_E_P_N/Polyps2'
apply_transformations(folder_path)
