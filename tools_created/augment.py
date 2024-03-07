import albumentations as A
import cv2
import os 
from matplotlib import pyplot as plt

"""
This script takes a directory of images and applies the augmentations specified with the
different arrays. 
The augmentation will be stored in the same directory as the normal images
"""

# Define the augmentations for polyp class: 
# when rotating the dimension is kept -> distortion (maybe do it different way)
rotate90 = A.SafeRotate(limit = (90, 90), p = 1.0, border_mode = cv2.BORDER_CONSTANT, value = 0)
rotate180 = A.SafeRotate(limit = (180, 180), p = 1.0)
rotate270 = A.SafeRotate(limit = (270, 270), p = 1.0, border_mode = cv2.BORDER_CONSTANT, value = 0)
rotations = [rotate90, rotate180, rotate270]

brightness1 = A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0.2, 0.2), p=1.0)
brightness2 = A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=(-0.2, -0.2), p=1.0)
brightness = [brightness1, brightness2]


elastic1 = A.ElasticTransform(alpha=1, sigma=50, alpha_affine=80, border_mode = cv2.BORDER_CONSTANT, value = 0, p=1.0)
elastic = [elastic1]
# only do it for 0 degree and 180 degree rotations
apply_elastic = [0, 2]

# maybe set fit_output to true, is quite similar to elastic1
perspective1 = A.Perspective(scale=(0.2, 0.2), fit_output = False, p=1.0)
perspective = [perspective1]
# only do it for 90 degree and 270 degree rotations 
apply_perspective = [1, 3]

# this defines all augmentations
# every rotation and every brightness specified in rotations and brightness array are done to the base images
# if apply_elastic is not empty: then the elastic transformations specified will 
# be done for each image in image_transformations at each index in apply_elastic.
# the same principle goes for apply_perspective 
polyp_augmentations = {'rotations': rotations, 'brightness': brightness, 'apply_elastic': apply_elastic, 'elastic': elastic, 'apply_perspective': apply_perspective, 'perspective': perspective}


# Define the augmentations for barretts class:
apply_perspective = []
barretts_augmentations = {'rotations': rotations, 'brightness': brightness, 'apply_elastic': apply_elastic, 'elastic': elastic, 'apply_perspective': apply_perspective, 'perspective': perspective}

# Define the augmentations for esophagitis class: 
rotations = []
brightness = []
apply_elastic = [0]
apply_perspective = []
esophagitis_augmentations = {'rotations': rotations, 'brightness': brightness, 'apply_elastic': apply_elastic, 'elastic': elastic, 'apply_perspective': apply_perspective, 'perspective': perspective}

def apply_transformations(folder_path, augmentations):

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
            for rot in augmentations['rotations']: 
                transformed = rot(image = image)
                image_transformations.append(transformed['image'])

            for bright in augmentations['brightness']:
                transformed = bright(image = image)
                image_transformations.append(transformed['image'])

            for idx in augmentations['apply_elastic']: 
                for elastic_aug in augmentations['elastic']:
                    transformed = elastic_aug(image = image_transformations[idx])
                    image_transformations.append(transformed['image'])

            for idx in augmentations['apply_perspective']: 
                for perspec_aug in augmentations['perspective']:
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
folder_path = '../../B_E_P_N/polyps2'
apply_transformations(folder_path, polyp_augmentations)

folder_path = '../../B_E_P_N/barretts2'
apply_transformations(folder_path, barretts_augmentations)

folder_path = '../../B_E_P_N/esophagitis2'
apply_transformations(folder_path, esophagitis_augmentations)



print('Your images are augmented')
