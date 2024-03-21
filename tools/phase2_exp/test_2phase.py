
from mmpretrain import ImageClassificationInferencer
import torch 

# for classification of healthy or unhealthy
model1, model2 = 'efficientnet_b4', 'efficientnet_b4'
schedule1 = '0.001', '0.001'
epoch1 = '100'
epoch2 = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]



ymodel1_config = f'../../config/phase2/{model1}_healthy.py'
model1_pretrained = f'../../work_dirs/phase2/{model1}/healthy/{schedule1}/epoch_{epoch1}.pth'
txt_path=f"../../work_dirs/phase2/{model1}/test/2phase.txt"

model2_config = f'../../config/phase2/{model2}_disease.py'

cfg = Config.fromfile(model1_config)

model1 = ImageClassificationInferencer(model = model1_config, weight = model1_pretrained)

paths1, paths2 = [], []
labels1, labels2 = [], []

tp = torch.empty(len(epoch2), 4)
fp = torch.empty(len(epoch2), 4)
fn = torch.empty(len(epoch2), 4)
tn = torch.empty(len(epoch2), 4)

with open("../../../B_E_P_N/meta/test.txt", "r") as file:
    for line in file:
        path, label = line.strip().split(" ", 1)
        paths.append(f'../{path}')
        labels.append(label)

for label, path in zip(labels1, paths1):
    res = model1(path)
    if res['pred_class'] == 1:
        paths2.append(path)
        labels2.append(label)
    if label == res['pred_class'] and label == 0:
        tp[:][label] += 1
    elif res['pred_class'] != label and label >= 1:
        fp[:][res['pred_class']] += 1
        
        
# for classification of the concrete disease
for i, epoch in enumerate(epoch2): 
    model2_pretrained = f'../../work_dirs/phase2/{model2}/disease/{schedule2}/epoch_{epoch}.pth'
    model2 = ImageClassificationInferencer(model = model2_config, weight = model2_pretrained)

    for label, path in zip(labels2, paths2):
        res = model2(path) 
        if res['pred_class'] == label:
            tp[i][label] += 1
        else:
            fn[i][label] += 1
            fp[i][res['pred_class']] += 1

recall_mean = torch.mean(tp / (tp + fn), dim = 0) 
precision_mean = torch.mean(tp / (tp + fp), dim = 0)
accuracy_mean = torch.mean(torch.sum(tp, dim = 1) / len(paths1), dim = 0)
f1_mean = torch.mean(2 * tp / (tp + fn) * tp / (tp + fp) / (tp / (tp + fn) + tp / (tp + fp)), dim = 0)

recall_std = torch.std(tp / (tp + fn), dim = 0) 
precision_std = torch.std(tp / (tp + fp), dim = 0)
accuracy_std = torch.std(torch.sum(tp, dim = 1) / len(paths1), dim = 0)
f1_std = torch.std(2 * tp / (tp + fn) * tp / (tp + fp) / (tp / (tp + fn) + tp / (tp + fp)), dim = 0)


with open(txt_path, 'a') as file:
    algorithm = 'Two-Phase'
    file.write(f"\n\nAlgorithm: {algorithm} with Model: {model} with schedule: {key} \n")

    metrics = ['Accuracy:', 'Classwise Recall:', 'Classwise Precision:', 'Classwise F1-Score:']
    tensors_mean = [accuracy_mean, recall_mean, precision_mean, f1_mean]
    tensors_std = [accuracy_std, recall_std, precision_std, f1_std]

    for metric, tensor_mean, tensor_std in zip(metrics, tensors_mean, tensors_std):
        tensor_mean = np.round(tensor_mean.cpu().numpy(), 4)
        tensor_std = np.round(tensor_std.cpu().numpy(), 4)
        file.write(f"{metric} \n mean: \t {tensor_mean} \n std: \t {tensor_std} \n\n")










