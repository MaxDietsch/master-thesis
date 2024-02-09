#from models.densenet import DenseNet121
import torch
from dataset import Gastro_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.densenet import DenseNet121


def get_overloaded_samples(k):
    model.eval()
    for batch_index, (image, label) in enumerate(train_dataloader):
        if 1 in label:
            print(batch_index)
            

def train(epoch):
    model.train()
    for batch_index, (image, label) in enumerate(train_dataloader):
        image = image.to(device)
        label = label.to(device)
        # zero out gradients
        optimier.zero_grad()
        get_overloaded_samples(k)





# data preprocessing
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
    ])


# prepare the dataset
dataset_name = 'B_E_P_N-without-mix'
training_data = Gastro_Dataset(
        annotations_file = f'../../{dataset_name}/meta/train.txt',
        img_dir = f'../../{dataset_name}/train',
        transform = transform
        )

# define dataloader
batch_size = 2
train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)


# load the model with pretrained weights
model = DenseNet121()
weights_path = '../work_dirs/densetnet121_sgd_bepnwom_default/epoch_400.pth'
model.load_state_dict(torch.load(weights_path)['state_dict'])

# use gpu if available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('CUDA is available')
else: 
    device = torch.device('cpu')
    print('CUDA is not available')



# optimize
lr = 0.05
milestones = [10, 20, 30]
gamma = 0.2
weight_decay = 0.001 
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma)
iter_per_epoch = len(train_dataloader)

get_overloaded_samples(1)
get_overloaded_samples(1)
get_overloaded_samples(1)
get_overloaded_samples(1)



epochs = 40
save_dir = './work-dir'
for epoch in range(1, epochs + 1):
    train(epoch)
    evaluation()
    torch.save(model.state_dict(), os.path.join(save_dir, 'ck_{}.pth'.format(epoch)))












inputs, labels = next(iter(train_dataloader))
outputs = model(inputs)

print(outputs)
print(labels)
