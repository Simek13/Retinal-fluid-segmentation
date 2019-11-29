from os import listdir
from PIL import Image
from os.path import join


def calculate_class_weights_presence(root_path):
    masks_path = root_path + "/masks"
    masks = [d for d in listdir(masks_path)]
    class_counter = [0, 0, 0, 0]

    for m in masks:
        mask = Image.open(join(root_path, masks, m))
        pixels = mask.getdata()
        if 0 in pixels:
            class_counter[0] += 1
        if 85 in pixels:
            class_counter[1] += 1
        if 170 in pixels:
            class_counter[2] += 1
        if 255 in pixels:
            class_counter[3] += 1

    class_weights = {
        0: class_counter[0]/class_counter[0],
        1: class_counter[0]/class_counter[1],
        2: class_counter[0] / class_counter[2],
        3: class_counter[0] / class_counter[3]
    }
    return  class_weights


def calculate_class_weights_pixel_wise(root_path):
    masks_path = root_path + "/masks"
    masks = [d for d in listdir(masks_path)]
    class_counter = [0, 0, 0, 0]

    for m in masks:
        mask = Image.open(join(root_path, masks, m))
        pixels = mask.getdata()
        for p in pixels:
            if p == 0:
                class_counter[0] += 1
            if p == 85:
                class_counter[1] += 1
            if p == 170:
                class_counter[2] += 1
            if p == 255:
                class_counter[3] += 1

    class_weights = {
        0: class_counter[0] / class_counter[0],
        1: class_counter[0] / class_counter[1],
        2: class_counter[0] / class_counter[2],
        3: class_counter[0] / class_counter[3]
    }
    return class_weights
