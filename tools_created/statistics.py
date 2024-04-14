import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools

def analyze_image_resolutions(directory):
    # A dictionary to store resolution frequencies
    resolutions = {}

    # Walk through the directory
    for file in os.listdir(directory):
        if file.lower().endswith('.jpg'):
            image_path = os.path.join(directory file)
            image = cv2.imread(image_path)
            resolutions[(img.shape[1], img.shape[0])] += 1           

    print(resolutions)
    widths, heights, counts = zip(*[(w, h, c) for (w, h), c in resolutions.items()])

    unique_widths = sorted(set(widths))
    unique_heights = sorted(set(heights))

    # Initialize a matrix to hold the counts for each width-height combination
    count_matrix = np.zeros((len(unique_heights), len(unique_widths)))

    # Populate the count matrix
    for (w, h), count in resolution_stats.items():
    width_index = unique_widths.index(w)
    height_index = unique_heights.index(h)
    count_matrix[height_index, width_index] = count


    tick_labels_x = unique_heights
    tick_labels_y = unique_widths
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.colorbar()
    tick_marks_x = unique_widths
    tich_marks_y = unique_heights
    plt.xticks(tick_marks_x, tick_labels_x, rotation=45)
    plt.yticks(tick_marks_y, tick_labels_y)

    fmt = ".4f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="black" if cm[i, j] > thresh else "white")
  
    plt.tight_layout(pad = 2)
    plt.ylabel("Height")
    plt.xlabel("Width")
    plt.grid(False)
    plt.savefig('resolution.png')
    plt.close()


analyze_image_resolutions('.')

