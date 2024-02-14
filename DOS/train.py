import os
import torch
from dataset import Gastro_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.densenet import DenseNet121
from models.losses.f_loss import F_Loss
from models.losses.g_loss import G_Loss

# maybe make big variables global so that no need to pass them via functions
num_classes = 4
samples_per_class = [0, 44, 132, 539]
r = [0, 1, 2, 3]
z = {'image': [], 'n': [], 'w': []}

v = [[] for _ in range(num_classes)]
d = [torch.zeros((i, i)) for i in samples_per_class]

batch_idx = [[] for _ in range(num_classes)]
k = [0, 3, 3, 3]



def calc_mutual_distance_matrix():
    global d, v
    for h in range(num_classes):
        for i in range(samples_per_class[h]):
            for j in range(i, samples_per_class[h]):
                # do it symmmetrically for runtime
                if i == j:
                    d[h][i, j] = 999999
                    continue
                d[h][i, j] = torch.norm(v[h][i] - v[h][j])  # Euclidean distance
                d[h][j, i] = d[h][i, j]


def generate_overloaded_samples():
    global v, batch_idx, n, r, z, d, k, device
    
    # set of all deep features
    with torch.no_grad():
        for batch_index, (image, label) in enumerate(dos_dataloader):
            # maybe change here to get different amount of classes
            
            image = image.to(device)
            v[label].append(model.neck(model.backbone(image))[0][0])

            #print(model.neck(model.backbone(image))[0][0])

            # to store where the image is located in the dataloader
            batch_idx[label].append(batch_index) 
    
    calc_mutual_distance_matrix()
    
    #print(d)
    for i in range(num_classes):
        for j in range(samples_per_class[i]):
            n = []

            #print(v[i][0:5])
            #print(d[i])
            #print(torch.topk(d[i][j], k[i], largest = False).indices[1 : ])

            for x in torch.topk(d[i][j], k[i], largest = False).indices[1 : ]:
                n.append(v[i][x])


            #print(n) 

            # sample the vectors 
            w = torch.abs(torch.randn(r[i], k[i]))
            w /= torch.norm(w, dim=1, keepdim=True)          
            
            z['image'].append(batch_idx[i][j])
            z['n'].append(n)
            z['w'].append(w)

            #print(z)
            #print('"' * 40)
    
    v = [[] for _ in range(num_classes)]
    batch_index = [[] for _ in range(num_classes)]
       


def train(epoch):
    global z, device
    model.train()

    generate_overloaded_samples()
    total_loss = 0
    for batch_idx, (image, label) in enumerate(dos_dataloader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        deep_feat = model.neck(model.backbone(image))

        # calculate l_f
        f_loss = 0
        for w in z['w'][0]:
            x = z['n'][batch_idx]
            f_loss += f(deep_feat[0], z['n'][batch_idx], w)

        # calculate l_g
        g_loss = 0
        for w in z['w'][0]:
            classification = model.head(n).unsqueeze(0)
            g_loss += g(deep_feat[0], classification, label, z['n'][batch_idx], w)
        
        loss = g_loss + f_loss
        total_loss += loss
        loss.backward()
        optimizer.step()
    print(f'Training Epoch: {epoch} \t Average Loss: {total_loss.item() / len(dos_dataloader)} \t  Allocated Memory on GPU: {torch.cuda.memory_allocated() / (1024 ** 3)} GB')



def evaluation():
    model.eval()
    tp = [0 for _ in range(num_classes)]
    fp = [0 for _ in range(num_classes)]
    fn = [0 for _ in range(num_classes)]
    tn = [0 for _ in range(num_classes)]
    for (image, label) in eval_dataloader:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            predicted_class = torch.argmax(model(image))
            print(model(image))
            print(predicted_class)
            print(label)
        if predicted_class == label:
            tp[label] += 1
            tn = [elem + 1 if idx != label else elem for idx, elem in enumerate(tn)]
        else:
            fp[predicted_class] += 1
            fn[label] += 1
    
    precision = [(tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) != 0 else 0) for i in range(num_classes)]
    recall = [(tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) != 0 else 0) for i in range(num_classes)]
    f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0 for i in range(num_classes)]
    accuracy = ((sum(tp) + sum(tn)) / (sum(tp) + sum(fp) + sum(fn) + sum(tn)))

    print(f'Classwise Precision: \t {precision} \nClasswise Recall: \t {recall} \nClasswise F1-Score: \t {f1} \nAccuracy: \t {accuracy}')


# data preprocessing
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [151.78, 103.29, 97.41], std = [69.92, 55.96, 54.84])
    ])

# prepare the dataset for calculating the deep features, (only minority classes are needed
# in my case) has to be created in a clever way
dos_dataset_name = 'B_E_P_N-without-mix'
dos_data = Gastro_Dataset(
        annotations_file = f'../../{dos_dataset_name}/meta_min/train.txt',
        img_dir = f'../../{dos_dataset_name}/train',
        transform = transform
        )

# define dos dataloader
dos_batch_size = 1
dos_dataloader = DataLoader(dos_data, batch_size = dos_batch_size, shuffle = False)



# prepare the dataset 

eval_dataset_name = 'B_E_P_N-without-mix'
eval_data = Gastro_Dataset(
        annotations_file = f'../../{eval_dataset_name}/meta/val.txt',
        img_dir = f'../../{eval_dataset_name}/val',
        transform = transform
        )

# define dataloader
batch_size = 1
eval_dataloader = DataLoader(eval_data, batch_size = batch_size, shuffle = True)


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
f = F_Loss()
g = G_Loss()


model = model.to(device)
epochs = 40
save_dir = './work-dir'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_epoch = []


print("Start Training:")
for epoch in range(1, epochs + 1):
    #train(epoch)
    evaluation()
    if epoch in save_epoch:
        print(f'Save model at epoch: {epoch}')
        torch.save(model.state_dict(), os.path.join(save_dir, 'ck_{}.pth'.format(epoch)))
    print('\n')




