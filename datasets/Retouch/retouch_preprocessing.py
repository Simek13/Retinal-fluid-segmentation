from PIL import Image, ImageFile
from os import listdir, mkdir
from os.path import join, splitext
import numpy as np
from itertools import groupby
import shutil
import random
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


def print_current(text):
    print('###################')
    print(text)
    print('###################')


def intensity_process(masks_path):
    masks = listdir(masks_path)
    for m in masks:
        mask = np.array(Image.open(join(masks_path, m)))
        mask[mask == 85] = 1
        mask[mask == 170] = 2
        mask[mask == 255] = 3
        mask = Image.fromarray(mask)
        mask.save(join(masks_path, m.replace('CIRRUS', 'Cirrus')))


def volume_split(root_path, mask_ext='.png'):
    image_path = join(root_path, 'images')
    mask_path = join(root_path, 'masks')
    images = listdir(image_path)
    image_volumes = groupby(sorted(images), key=lambda x: x.split('_')[1])
    for gr, items in image_volumes:
        items = list(items)
        i = items[0]
        volume_id = i.split('_')[1]
        volume_path = join(root_path, 'Volume_' + volume_id)
        volume_images_path = join(volume_path, 'images')
        volume_masks_path = join(volume_path, 'masks')
        try:
            mkdir(volume_path)
        except:
            pass
        try:
            mkdir(volume_images_path)
        except:
            pass
        try:
            mkdir(volume_masks_path)
        except:
            pass
        for i in items:
            mask_name = splitext(i)[0] + mask_ext
            shutil.copyfile(join(image_path, i), join(volume_images_path, i))
            shutil.copyfile(join(mask_path, mask_name), join(volume_masks_path, mask_name))


def train_val_test_split(root_path):
    dirs = listdir(root_path)
    volume_dirs = [x for x in dirs if x.startswith('Volume')]
    random.shuffle(volume_dirs)
    return volume_dirs[:16], volume_dirs[16:20], volume_dirs[20:]


def copy_volumes(root_path, train_volumes, val_volumes, test_volumes):
    train_path = join(root_path, 'train')
    val_path = join(root_path, 'val')
    test_path = join(root_path, 'test')
    try:
        shutil.rmtree(train_path)
    except:
        pass
    try:
        shutil.rmtree(val_path)
    except:
        pass
    try:
        shutil.rmtree(test_path)
    except:
        pass
    for v in train_volumes:
        copy_tree(join(root_path, v), join(train_path, v))
    for v in val_volumes:
        copy_tree(join(root_path, v), join(val_path, v))
    for v in test_volumes:
        copy_tree(join(root_path, v), join(test_path, v))


def volume_class_distribution_analysis(root_path, volumes):
    im_counter = 0
    fluid_appearances = np.zeros(3)
    for v in volumes:
        masks_dir = join(root_path, v, 'masks')
        masks = listdir(masks_dir)
        im_counter += len(masks)
        for m in masks:
            mask = np.array(Image.open(join(masks_dir, m)))
            for i in range(fluid_appearances.size):
                if i + 1 in mask:
                    fluid_appearances[i] += 1
    return (fluid_appearances / im_counter) * 100


def v_class_distribution_analysis(root_path, volumes):
    im_counter = 0
    fluid_appearances = np.zeros(3)
    for v in volumes:
        masks_dir = join(root_path, v, 'masks')
        masks = listdir(masks_dir)
        im_counter += 1
        f_appearances = np.zeros(3)
        for m in masks:
            mask = np.array(Image.open(join(masks_dir, m)))
            for i in range(fluid_appearances.size):
                if f_appearances[i] < 1:
                    if i + 1 in mask:
                        f_appearances[i] += 1
            if all(x>0 for x in f_appearances):
                break
        fluid_appearances += f_appearances
    return (fluid_appearances / im_counter) * 100


def check_class_distr(train, val, test):
    return abs(train - val) < 11 and abs(train - test) < 11 and abs(val - test) < 11


if __name__ == '__main__':
    print_current('Starting preprocessing...')
    root_path = './Spectralis'
    # volume_split(root_path)
    while True:
        train_vols, val_vols, test_vols = train_val_test_split(root_path)
        train_dist = v_class_distribution_analysis(root_path, train_vols).tolist()
        val_dist = v_class_distribution_analysis(root_path, val_vols).tolist()
        test_dist = v_class_distribution_analysis(root_path, test_vols).tolist()
        if check_class_distr(train_dist[0], val_dist[0], test_dist[0]) and \
                check_class_distr(train_dist[1], val_dist[1], test_dist[1]) and \
                check_class_distr(train_dist[2], val_dist[2], test_dist[2]):
            break
    print(train_dist)
    print(val_dist)
    print(test_dist)
    copy_volumes(root_path, train_vols, val_vols, test_vols)
    X = np.arange(3)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(X + 0.00, train_dist, color='b', width=0.25)
    ax.bar(X + 0.25, val_dist, color='g', width=0.25)
    ax.bar(X + 0.50, test_dist, color='r', width=0.25)
    ax.set_ylabel('%')
    ax.set_xticks(X + 0.25)
    ax.set_xticklabels(('IRF', 'SRF', 'PED'))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(labels=['train', 'val', 'test'])
    plt.savefig(join(root_path, 'class_distribution' + '_spectralis' + '.png'),
                format='png', bbox_inches='tight')

    # process(spectralis_masks)
    print_current('Preprocessing done')
