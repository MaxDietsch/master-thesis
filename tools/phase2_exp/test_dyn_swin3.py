from mmpretrain import ImageClassificationInferencer
import torch
from mmengine.config import Config
import numpy as np

# for classification of healthy or unhealthy
model1_name, model2_name = 'swin', 'swin'
schedule1, schedule2 = 'lr_0.01', 'lr_decr'
epoch1 = '100'
epoch2 = [51, 52, 53, 54, 55, 56, 57, 58, 59]
model1_type = '_aug'
model2_type = '_aug'
model1_phase = 'phase2'



model1_config = f'../../config/{model1_phase}/{model1_name}{model1_type}.py'
model1_file = f'../../work_dirs/{model1_phase}/{model1_name}/{model1_type[1 : ]}/{schedule1}/epoch_{epoch1}.pth'
txt_path=f"../../work_dirs/phase2/{model1_name}/test/dyn{model2_type}.txt"

model2_config = f'../../config/phase2/{model2_name}_dyn{model2_type}.py'

model1 = ImageClassificationInferencer(model = model1_config, pretrained = model1_file)


paths1, labels1 = [], []
with open("../../../B_E_P_N/meta/test.txt", "r") as file:
    for line in file:
        path, label = line.strip().split(" ", 1)
        paths1.append(f'../{path}')
        labels1.append(int(label))

res_arr = []
for label, path in zip(labels1, paths1):
    res_arr.append(model1(path)[0])

tp = torch.zeros(len(epoch2), 4)
fp = torch.zeros(len(epoch2), 4)
fn = torch.zeros(len(epoch2), 4)
tn = torch.zeros(len(epoch2), 4)


for i, epoch in enumerate(epoch2): 
    model2_pretrained = f'../../work_dirs/phase2/{model2_name}/dyn{model2_type}/{schedule2}/epoch_{epoch}.pth'
    model2 = ImageClassificationInferencer(model = model2_config, pretrained = model2_pretrained)
    
    j = 0
    for label, path in zip(labels1, paths1):
        res1 = res_arr[j]
        j += 1
        res2 = model2(path)[0]

        if res1['pred_label'] == 0 and res2['pred_label'] == 0:
            res = 0
        if res1['pred_label'] != 0:
            res = res2['pred_label']
        if res1['pred_label'] == 0 and res2['pred_label'] != 0:
            pred_score1 = res1['pred_score']
            pred_score2 = res2['pred_score']
            if pred_score2 >= pred_score1:
                res = res2['pred_label']
            else:
                res = res1['pred_label']


        if res == label:
            tp[i][label] += 1
        else:
            fn[i][label] += 1
            fp[i][res] += 1


recall_epochs = tp / (tp + fn) * 100
precision_epochs = tp / (tp + fp) * 100
accuracy_epochs = torch.sum(tp, dim = 1) / len(paths1) * 100

recall_epochs[torch.isnan(recall_epochs)] = 0
precision_epochs[torch.isnan(precision_epochs)] = 0
accuracy_epochs[torch.isnan(accuracy_epochs)] = 0

f1_epochs = torch.zeros(recall_epochs.shape)
for i in range(len(epoch2)):
    for j in range(4):
        f1_epochs[i][j] = 2 * recall_epochs[i][j] * precision_epochs[i][j] / (recall_epochs[i][j] + precision_epochs[i][j]) if recall_epochs[i][j] + precision_epochs[i][j] != 0 else 0


print(recall_epochs)
print(precision_epochs) 
print(accuracy_epochs)
print(f1_epochs)

recall_mean = torch.mean(recall_epochs, dim = 0) 
precision_mean = torch.mean(precision_epochs, dim = 0)
accuracy_mean = torch.mean(accuracy_epochs, dim = 0)
f1_mean = torch.mean(f1_epochs, dim = 0)


recall_std = torch.std(recall_epochs, dim = 0) 
precision_std = torch.std(precision_epochs, dim = 0)
accuracy_std = torch.std(accuracy_epochs, dim = 0)
f1_std = torch.std(f1_epochs, dim = 0)


with open(txt_path, 'a') as file:
    algorithm = 'Dynamic Sampling'
    file.write(f"\n\nAlgorithm: {algorithm} with Model: {model2_name} with schedule: {schedule2} \n")

    metrics = ['Accuracy:', 'Classwise Recall:', 'Classwise Precision:', 'Classwise F1-Score:']
    tensors_mean = [accuracy_mean, recall_mean, precision_mean, f1_mean]
    tensors_std = [accuracy_std, recall_std, precision_std, f1_std]

    for metric, tensor_mean, tensor_std in zip(metrics, tensors_mean, tensors_std):
        tensor_mean = np.round(tensor_mean.cpu().numpy(), 4)
        tensor_std = np.round(tensor_std.cpu().numpy(), 4)
        file.write(f"{metric} \n mean: \t {tensor_mean} \n std: \t {tensor_std} \n\n")


