import torch
import cv2
from torchvision import transforms
import torch.nn as nn
"""
This file takes in a dataset (so the information about it). It uses the balancedMixUp strategy proposed in: 
Balanced-MixUp for Highly Imbalanced Medical Image Classification by Galdran et al. 
to create new samples and add them to a new dataset (the old files will also be copied to the new data directory)
The txt which is read should have the following structure: pathToFile label -> ../../B_E_P_N/train/polyp1.png 1
"""

img_dir = '../../B_E_P_N_BMU/train'
train_txt_file = '../../B_E_P_N_BMU/meta/train.txt'
class_distribution = torch.tensor([3312, 45, 132, 539])
num_classes = len(class_distribution)
alpha = 0.5
num_new_samples = 10
torch.manual_seed(16)

paths = [[] for _ in range(num_classes)]

with open(train_txt_file, "r") as file:  
    for line in file:
        path, label = line.strip().split(" ", 1)
        paths[int(label)].append(path)


# define categorical distribution for instance based sampling 
prob_is = class_distribution / torch.sum(class_distribution, dim = 0)
cat_dist = torch.distributions.categorical.Categorical(probs = prob_is)

# beta distribution for sampling lambda
beta_dist = torch.distributions.beta.Beta(alpha, 1)

for i in range(num_new_samples):
    
    # class based sampling
    sc = torch.randint(low = 0, high = num_classes, size = (1, ))
    img_sc = paths[sc][torch.randint(low = 0, high = len(paths[sc]), size = (1, ))]
    img_sc = cv2.imread(img_sc)
    img_sc = cv2.cvtColor(img_sc, cv2.COLOR_BGR2RGB)
    img_sc = transforms.ToTensor()(img_sc)

    # instance based sampling and reshape image to match dimension of the first image 
    si = cat_dist.sample(sample_shape=(1,))
    img_si = paths[si][torch.randint(low = 0, high = len(paths[si]), size = (1, ))]
    img_si = cv2.imread(img_si)
    img_si = cv2.cvtColor(img_si, cv2.COLOR_BGR2RGB)
    img_si = transforms.ToTensor()(img_si)
    img_si = img_si.unsqueeze(0)
    img_si = img_si.unsqueeze(0)
    img_si = nn.functional.interpolate(img_si, size = img_sc.shape, mode = 'trilinear')
    img_si = img_si.squeeze()
    print(img_si)
    # sample lambda
    l = beta_dist.sample()

    # mix the sample
    mixed_sample = l * img_si + (1 - l) * img_sc
    mixed_label = l * si + (1 - l) * sc

    mixed_image = mixed_sample.cpu().numpy().transpose(1, 2, 0)
    print(mixed_image)
    mixed_image = cv2.cvtColor(mixed_image, cv2.COLOR_RGB2BGR)  
    cv2.imwrite(img_dir + f'/bmu_{i}.jpg', mixed_image) 

print('The images are created. You can start to train!')






    





