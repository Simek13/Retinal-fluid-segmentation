from PIL import Image
import os
import numpy as np


def print_current(text):
    print('###################')
    print(text)
    print('###################')


def process(masks_path):
    masks = os.listdir(masks_path)
    for m in masks:
        mask = np.array(Image.open(os.path.join(masks_path, m)))
        mask[mask == 85] = 1
        mask[mask == 170] = 2
        mask[mask == 255] = 3
        mask = Image.fromarray(mask)
        mask.save(os.path.join(masks_path, m))


if __name__ == '__main__':
    print_current('Starting preprocessing...')
    cirrus_masks = './Cirrus/masks'
    spectralis_masks = './Spectralis/masks'
    process(cirrus_masks)
    process(spectralis_masks)
    print_current('Preprocessing done')

