import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, class_names, normalize=True, filename="confusion_matrix.png", cmap="viridis"):
  """
  Plots a confusion matrix with labels and saves it as an image.

  Args:
      cm (torch.Tensor): Confusion matrix tensor.
      normalize (bool, optional): Normalize values to percentages. Defaults to True.
      filename (str, optional): Filename to save the image. Defaults to "confusion_matrix.png".
      cmap (str, optional): Colormap for the plot. Defaults to "Blues".
  """
  if normalize:
    cm_sum = cm.sum(axis=1, keepdim=True)
    cm = cm / cm_sum
    plt.title(f"Normalized Confusion Matrix \n of {model} with schedule {schedule} \n used: aug1, ros25, transfer learning, CoSen")
  else:
    plt.title(f"Confusion Matrix of {model} with schedule {schedule}")

  classes = range(cm.shape[0])
  tick_labels = class_names
  #plt.figure(figsize = (12, 8))
  plt.imshow(cm, interpolation="nearest", cmap=cmap)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, tick_labels, rotation=45)
  plt.yticks(tick_marks, tick_labels)

  # Set text for cells
  fmt = ".4f" if normalize else "d"
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
             color="black" if cm[i, j] > thresh else "white")

  plt.tight_layout(pad = 2)
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.grid(False)
  plt.savefig(filename)
  plt.close()



# Read the PyTorch tensor from the file
model = 'EfficientNet-B4'
schedule = 'lr_decr'
filename = f"../work_dirs/phase3/efficientnet_b4/test/ros25_aug_pretrained_cosen/lr_decr/cm/avg_cm.pt"  # Replace with your actual filename
tensor = torch.load(filename)

# Ensure it's a square tensor with values between 0 and 1
print(tensor)

class_names = ['normal', 'polyps', 'barretts', 'esophagitis']

plot_confusion_matrix(tensor, class_names, normalize=True, filename=filename[ : -3 ] + '.png')

