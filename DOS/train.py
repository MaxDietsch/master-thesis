#from models.densenet import DenseNet121
import torch
from dataset import Gastro_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.densenet import DenseNet121
from models.losses.f_loss import F_Loss

# maybe make big variables global so that no need to pass them via functions
num_classes = 4
samples_per_class = [5, 5, 5, 5]
r = [0, 1, 2, 3]
z = {'image': [], 'n': [], 'w': []}
v = [[] for _ in range(num_classes)]
d = [torch.zeros((i, i)) for i in samples_per_class]
batch_idx = [[] for _ in range(num_classes)]


def calc_mutual_distance_matrix():
    global d, v
    for h in range(num_classes):
        for i in range(samples_per_class[h]):
            for j in range(samples_per_class[h]):
                print(v[h])
                print(v[h][i])
                print(v[h][i][0])
                print(v[h][i][0][0])
                d[h][i, j] = torch.norm(v[h][i][0] - v[h][j][0])  # Euclidean distance


def generate_overloaded_samples():
    global v, batch_idx, n, r, z, d
    
    # set of all deep features
    with torch.no_grad():
        for batch_index, (image, label) in enumerate(dos_dataloader):
            # maybe change here to get different amount of classes
            image.to(device)
            v[label].append(model.neck(model.backbone(image)))
            # to store where the image is located in the dataloader
            batch_idx[label].append(batch_index) 
    
    #
    print(v[0][0])
    calc_mutual_distance_matrix()
    print(d)
    for i in range(num_classes):
        for j in batch_idx[i]:
            n = v[label][torch.topk(d[i], k[i], largest = False)[1]]
            
            # sample the vectors 
            w = torch.randn(r[i], k[i])
            w /= torch.norm(w, dim=1, keepdim=True)          
            
            z['image'].append(j)
            z['n'].append(n)
            z['w'].append(w)
    
    v = [[] for _ in range(num_classes)]
    n = []
    batch_index = [[] for _ in range(num_classes)]
       

def train(epoch):
    model.train()

    generate_overloaded_samples()
    z = sorted(z, key = lambda x : x[0])
    print(z) 
    return 
    
    for batch_idx, (image, label) in enumerate(dos_dataloader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        deep_feat = model.neck(model.backbone(image))

        # calculate l_f
        f_loss = 0
        for w in z['w']:
            f_loss += F_Loss(deep_feat, z['n'][batch_idx], w)

        # calculate l_g
        g_loss = 0
        for w in z['w']:
            classification = model.head(n)
            g_loss += G_Loss(deep_feat, classification, label, z['n'][batch_idx], w)
        
        loss = g_loss + f_loss
        loss.backward()
        optimizer.step()
        print(f'Training Epoch: {epoch} \t Loss: {loss.item()}')

def evaluation():
    model.eval()
    tp = [0 for _ in range(num_classes)]
    fp = [0 for _ in range(num_classes)]
    fn = [0 for _ in range(num_classes)]
    tn = [0 for _ in range(num_classes)]
    for (image, label) in evalDataLoader:
        image.to(device)
        label.to(device)
        with torch.no_grad():
            predicted_class = torch.argmax(model(image))
        if predicted_class == label:
            tp[label] += 1
            tn = [elem + 1 if idx != label else elem for idx, elem in enumerate(tn)]
        else:
            fp[predicted_class] += 1
            fn[label] += 1

    precision = [tp[i] / (tp[i] + fp[i]) for i in range(num_classes)]
    recall = [tp[i] / (tp[i] + fn[i]) for i in range(num_classes)]
    f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for _ in range(num_classes)]
    accuracy = ((sum(tp) + sum(tn)) / (sum(tp) + sum(fp) + sum(fn) + sum(tn)))

    print(f'Classwise Precision: \t {precision} \n Classwise Recall: \t {recall} \n Classwise F1-Score: \t {f1} \n Accuracy: \t {accuracy}')


# data preprocessing
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
    ])

# prepare the dataset for calculating the deep features, (only minority classes are needed
# in my case) has to be created in a clever way
dos_dataset_name = 'B_E_P_N_minority'
dos_data = Gastro_Dataset(
        annotations_file = f'../../{dos_dataset_name}/meta/train.txt',
        img_dir = f'../../{dos_dataset_name}/train',
        transform = transform
        )

# define dos dataloader
dos_batch_size = 1
dos_dataloader = DataLoader(dos_data, batch_size = dos_batch_size, shuffle = False)



# prepare the dataset 
"""
dataset_name = 'B_E_P_N-without-mix'
training_data = Gastro_Dataset(
        annotations_file = f'../../{dataset_name}/meta/train.txt',
        img_dir = f'../../{dataset_name}/train',
        transform = transform
        )

# define dataloader
batch_size = 2
train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
"""

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
iter_per_epoch = len(dos_dataloader)


epochs = 40
save_dir = './work-dir'
for epoch in range(1, epochs + 1):
    train(epoch)
    evaluation()


torch.save(model.state_dict(), os.path.join(save_dir, 'ck_{}.pth'.format(epoch)))




