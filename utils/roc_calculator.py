from PIL import Image
from os.path import join
from os import listdir
import numpy as np


if __name__ == '__main__':
    cirrus_masks = '../datasets/Retouch/Cirrus/masks'
    spectralis_masks = '../datasets/Retouch/Spectralis/masks'
    counter = 0
    num_labels = 4
    label_stats = {i: [] for i in range(num_labels)}
    prediction_scores = {i: [] for i in range(num_labels)}

    for mask in listdir(spectralis_masks):
        mask_data = np.array(Image.open(join(spectralis_masks, mask)).resize((64, 64)), dtype=np.uint8).flatten()
        for i in range(num_labels):
            y = np.zeros(mask_data.shape)
            y[mask_data == i] = 1
            label_stats[i].append(y)
