
from mmpretrain import ImageClassificationInferencer
import torch
from mmengine.config import Config
import numpy as np

# for classification of healthy or unhealthy
model1_name, model2_name = 'efficientnet_b4', 'efficientnet_b4'
schedule1, schedule2 = 'lr_decr', 'lr_0.001'
epoch1 = '100'
epoch2 = [91, 92, 93]



model1_config = f'../../config/phase2/{model1_name}_healthy.py'
model1_pretrained = f'../../work_dirs/phase2/{model1_name}/healthy/{schedule1}/epoch_{epoch1}.pth'
txt_path=f"../../work_dirs/phase2/{model1_name}/test/2phase.txt"

model2_config = f'../../config/phase2/{model2_name}_disease.py'

cfg = Config.fromfile(model1_config)

model1 = ImageClassificationInferencer(model = model1_config, pretrained = model1_pretrained)

paths1, paths2 = [], []
labels1, labels2 = [], []

tp = torch.zeros(len(epoch2), 4)
fp = torch.zeros(len(epoch2), 4)
fn = torch.zeros(len(epoch2), 4)
tn = torch.zeros(len(epoch2), 4)

with open("../../../B_E_P_N/meta/test.txt", "r") as file:
    for line in file:
        path, label = line.strip().split(" ", 1)
        paths1.append(f'../{path}')
        labels1.append(int(label))

counts= 0
for label, path in zip(labels1, paths1):
    res = model1(path)[0]
    if res['pred_label'] == 1:
        paths2.append(path)
        labels2.append(label)
        if label == 1:
            counts+= 1
    if label > 1:
        label = 1
    if label == res['pred_label'] and label == 0:
        tp[ : ,label] += 1
    elif res['pred_label'] != label and label != 0:
        fp[ : ,res['pred_label']] += 1
print(tp)
print(fn)
print(fp)
print('-' * 100)
print(counts)


        
# for classification of the concrete disease
for i, epoch in enumerate(epoch2): 
    model2_pretrained = f'../../work_dirs/phase2/{model2_name}/disease/{schedule2}/epoch_{epoch}.pth'
    model2 = ImageClassificationInferencer(model = model2_config, pretrained = model2_pretrained)

    for label, path in zip(labels2, paths2):
        res = model2(path)[0]
        if res['pred_label'] + 1 == label:
            tp[i][label] += 1
        else:
            fn[i][label] += 1
            fp[i][res['pred_label'] + 1] += 1

print(tp)
print(fn)
print(fp)
print('-' * 100)


recall_epochs = tp / (tp + fn) * 100
precision_epochs = tp / (tp + fp) * 100
accuracy_epochs = torch.sum(tp, dim = 1) / len(paths1) * 100

print(recall_epochs)
print(precision_epochs) 
print(accuracy_epochs)

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
    algorithm = 'Two-Phase'
    file.write(f"\n\nAlgorithm: {algorithm} with Model: {model2_name} with schedule: {schedule2} \n")

    metrics = ['Accuracy:', 'Classwise Recall:', 'Classwise Precision:', 'Classwise F1-Score:']
    tensors_mean = [accuracy_mean, recall_mean, precision_mean, f1_mean]
    tensors_std = [accuracy_std, recall_std, precision_std, f1_std]

    for metric, tensor_mean, tensor_std in zip(metrics, tensors_mean, tensors_std):
        tensor_mean = np.round(tensor_mean.cpu().numpy(), 4)
        tensor_std = np.round(tensor_std.cpu().numpy(), 4)
        file.write(f"{metric} \n mean: \t {tensor_mean} \n std: \t {tensor_std} \n\n")










