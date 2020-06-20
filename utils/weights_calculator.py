from PIL import Image
import os
import numpy as np


def print_current(text):
    print('###################')
    print(text)
    print('###################')


def calc_weights_presence(masks_path, n_labels):
    masks = os.listdir(masks_path)
    class_counter = [0] * n_labels
    for m in masks:
        mask = np.array(Image.open(os.path.join(masks_path, m)))
        for i in range(n_labels):
            if i in mask:
                class_counter[i] += 1
    class_weights = np.asarray(class_counter) / max(class_counter)
    return class_weights


def calc_weights_presence_balanced(masks_path, n_labels):
    masks = os.listdir(masks_path)
    class_counter = [0] * n_labels
    for m in masks:
        mask = np.array(Image.open(os.path.join(masks_path, m)))
        if np.count_nonzero(mask) != 0:
            for i in range(n_labels):
                if i in mask:
                    class_counter[i] += 1
    class_weights = np.asarray(class_counter) / max(class_counter)
    return class_weights


def calc_weights_pixel_wise(masks_path, n_labels):
    masks = os.listdir(masks_path)
    class_counter = [0] * n_labels
    for m in masks:
        mask = np.array(Image.open(os.path.join(masks_path, m)))
        for i in range(n_labels):
            class_counter[i] += np.count_nonzero(mask == i)
    class_weights = np.asarray(class_counter) / max(class_counter)
    return class_weights


def calc_weights_pixel_wise_balanced(masks_path, n_labels):
    masks = os.listdir(masks_path)
    class_counter = [0] * n_labels
    for m in masks:
        mask = np.array(Image.open(os.path.join(masks_path, m)))
        if np.count_nonzero(mask) != 0:
            for i in range(n_labels):
                class_counter[i] += np.count_nonzero(mask == i)
    class_weights = np.asarray(class_counter) / max(class_counter)
    return class_weights


if __name__ == '__main__':
    print_current('Calculating...')
    cirrus_masks = '../datasets/Retouch/Cirrus/masks'
    spectralis_masks = '../datasets/Retouch/Spectralis/masks'
    # calc_weights(cirrus_masks)
    print_current(calc_weights_presence(cirrus_masks, 4))
