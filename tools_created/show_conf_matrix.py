import torch
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names, normalize=True, filename="confusion_matrix.png", cmap="Blues"):
  """
  Plots a confusion matrix with labels and saves it as an image.

  Args:
      cm (torch.Tensor): Confusion matrix tensor.
      normalize (bool, optional): Normalize values to percentages. Defaults to True.
      filename (str, optional): Filename to save the image. Defaults to "confusion_matrix.png".
      cmap (str, optional): Colormap for the plot. Defaults to "Blues".
  """
  if normalize:
    cm = cm.sum(axis=1, keepdim=True)
    cm = cm / cm.sum(axis=0, keepdim=True)
    plt.title("Confusion Matrix (Normalized)")
  else:
    plt.title("Confusion Matrix")

  classes = range(cm.shape[0])
  tick_labels = class_names
  plt.imshow(cm, interpolation="nearest", cmap=cmap)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, tick_labels, rotation=45)
  plt.yticks(tick_marks, tick_labels)

  # Set text for cells
  fmt = ".2f" if normalize else "d"
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.grid(False)
  plt.savefig(filename)
  plt.close()



# Read the PyTorch tensor from the file
model = 'resnet50'
schedule = 'lr_decr'
filename = f"../work_dirs/phase1/{model}/test/{schedule}_cm.pt"  # Replace with your actual filename
tensor = torch.load(filename)

# Ensure it's a square tensor with values between 0 and 1
if not (tensor.shape[0] == tensor.shape[1]) or not (tensor.min() >= 0 and tensor.max() <= 1):
  raise ValueError("Input tensor must be square and have values between 0 and 1.")

class_names = ['normal', 'polyps', 'barretts', 'esophagitis']

plot_confusion_matrix(tensor, class_names, normalize=True, filename=filename[ : -3 ] + '.png')

