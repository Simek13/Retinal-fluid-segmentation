'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for loading training, validation, and testing data into the models.
It is specifically designed to handle 3D single-channel medical data.
Modifications will be needed to train/test on normal 3-channel images.
'''

from __future__ import print_function

import threading
from os.path import join, basename, splitext
from os import mkdir, listdir
from glob import glob
import csv
from sklearn.model_selection import KFold
import numpy as np
from numpy.random import shuffle
import SimpleITK as sitk
from tqdm import tqdm

import matplotlib

from preprocessing.intensity import normalize_robust

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()

from tensorflow.keras.preprocessing.image import *

from custom_data_aug import elastic_transform, salt_pepper_noise

from PIL import Image

debug = 1


def load_data(root):
    datasets_path = [root + "/imgs", root + "/masks"]
    image_data = [d for d in listdir(datasets_path[0])]
    # mask_data = [d for d in os.listdir(datasets_path[1])]
    data = []

    for im in image_data:
        name = splitext(im)[0]
        # data.append((name + '.jpg', name.replace('Cirrus', 'CIRRUS') + '.png'))
        data.append((name + '.jpg', name + '.png'))

    # train_data, test_data = train_test_split(data, test_size=0.1)

    return data


def compute_class_weights(root, train_data_list):
    '''
        We want to weight the the positive pixels by the ratio of negative to positive.
        Three scenarios:
            1. Equal classes. neg/pos ~ 1. Standard binary cross-entropy
            2. Many more negative examples. The network will learn to always output negative. In this way we want to
               increase the punishment for getting a positive wrong that way it will want to put positive more
            3. Many more positive examples. We weight the positive value less so that negatives have a chance.
    '''
    pos = 0.0
    neg = 0.0
    for img_name in tqdm(train_data_list):
        img = sitk.GetArrayFromImage(sitk.ReadImage(join(root, 'masks', img_name[1])))
        for slic in img:
            if not np.any(slic):
                continue
            else:
                p = np.count_nonzero(slic)
                pos += p
                neg += (slic.size - p)

    return neg / pos


def load_class_weights(root, split):
    class_weight_dir = join(root, 'split_lists')
    try:
        mkdir(class_weight_dir)
    except:
        pass
    class_weight_filename = join(class_weight_dir, 'train_split_' + str(split) + '_class_weights.npy')
    try:
        return np.load(class_weight_filename)
    except:
        print('Class weight file {} not found.\nComputing class weights now. This may take '
              'some time.'.format(class_weight_filename))
        train_data_list, _ = load_data(root)
        value = compute_class_weights(root, train_data_list)
        np.save(class_weight_filename, value)

        print('Finished computing class weights. This value has been saved for this training split.')
        return value


def split_data(root_path, num_splits=4):
    mask_list = []
    for ext in ('*.mhd', '*.hdr', '*.nii'):
        mask_list.extend(sorted(glob(join(root_path, 'masks', ext))))

    assert len(mask_list) != 0, 'Unable to find any files in {}'.format(join(root_path, 'masks'))

    outdir = join(root_path, 'split_lists')
    try:
        mkdir(outdir)
    except:
        pass

    kf = KFold(n_splits=num_splits)
    n = 0
    for train_index, test_index in kf.split(mask_list):
        with open(join(outdir, 'train_split_' + str(n) + '.csv'), 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in train_index:
                writer.writerow([basename(mask_list[i])])
        with open(join(outdir, 'test_split_' + str(n) + '.csv'), 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in test_index:
                writer.writerow([basename(mask_list[i])])
        n += 1


def convert_data_to_numpy(root_path, img_name, net_input_shape, no_masks=False, mask_name='', overwrite=False):
    fname = splitext(img_name)[0]
    numpy_path = join(root_path, 'np_files')
    img_path = join(root_path, 'images')
    mask_path = join(root_path, 'masks')
    fig_path = join(root_path, 'figs')
    try:
        mkdir(numpy_path)
    except:
        pass
    try:
        mkdir(fig_path)
    except:
        pass

    if not overwrite:
        try:
            with np.load(join(numpy_path, fname + '.npz')) as data:
                return data['img'], data['mask']
        except:
            pass

    try:
        # itk_img = sitk.ReadImage(join(img_path, img_name))
        # img = sitk.GetArrayFromImage(itk_img)
        # img = img.astype(np.float32)

        img = np.array(Image.open(join(img_path, img_name)).convert('L').resize((net_input_shape[1],
                                                                                 net_input_shape[0])),
                       dtype=np.float64)
        img = normalize_robust(img)
        img = np.clip(img, -1.0, 1.0)
        # img -= 127.5
        # img /= 127.5
        # print('Min: {}  Max: {}'.format(np.amin(img), np.amax(img)))
        # img = np.array(Image.open(join(img_path, img_name)).resize((net_input_shape[1], net_input_shape[0])),
        #                dtype=np.uint8)

        if not no_masks:
            # itk_mask = sitk.ReadImage(join(mask_path, mask_name))
            # mask = sitk.GetArrayFromImage(itk_mask)
            # mask = mask.astype(np.uint8)

            mask = np.array(Image.open(join(mask_path, mask_name)).resize((net_input_shape[1],
                                                                           net_input_shape[0])),
                            dtype=np.uint8)
            # mask = np.array(Image.open(join(mask_path, mask_name)),
            #                 dtype=np.uint8)

        try:

            plt.imshow(img, cmap='gray')
            if not no_masks:
                plt.imshow(mask, alpha=0.15)
                plt.show()

            fig = plt.gcf()
            fig.suptitle(fname)

            plt.savefig(join(fig_path, fname + '.png'), format='png', bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print('\n' + '-' * 100)
            print('Error creating qualitative figure for {}'.format(fname))
            print(e)
            print('-' * 100 + '\n')

        if not no_masks:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img, mask=mask)
        else:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img)

        if not no_masks:
            return img, mask
        else:
            return img

    except Exception as e:
        print('\n' + '-' * 100)
        print('Unable to load img or masks for {}'.format(fname))
        print(e)
        print('Skipping file')
        print('-' * 100 + '\n')

        return np.zeros(1), np.zeros(1)


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def augmentImages(batch_of_images, batch_of_masks):
    for i in range(len(batch_of_images)):
        img_and_mask = np.concatenate((batch_of_images[i, ...], batch_of_masks[i, ...]), axis=2)
        if img_and_mask.ndim == 4:  # This assumes single channel data. For multi-channel you'll need
            # change this to put all channel in slices channel
            orig_shape = img_and_mask.shape
            img_and_mask = img_and_mask.reshape((img_and_mask.shape[0:3]))

        # if np.random.randint(0, 10) == 7:
        #     img_and_mask = random_shift(img_and_mask, wrg=0.2, hrg=0.2, row_axis=0, col_axis=1, channel_axis=2,
        #                                 fill_mode='constant', cval=0.)

        # if np.random.randint(0, 10) == 7:
        # img_and_mask = random_rotation(img_and_mask, rg=20, row_axis=0, col_axis=1, channel_axis=2,
        #                                fill_mode='constant', cval=0.)

        # if np.random.randint(0, 10) == 7:
        #     img_and_mask = random_zoom(img_and_mask, zoom_range=(0.75, 0.75), row_axis=0, col_axis=1,
        #                                channel_axis=2,
        #                                fill_mode='constant', cval=0.)

        if np.random.randint(0, 5) == 3:
            img_and_mask = elastic_transform(img_and_mask, alpha=1000, sigma=80, alpha_affine=50)

        # if np.random.randint(0, 10) == 7:
        #     img_and_mask = random_shear(img_and_mask, intensity=16, row_axis=0, col_axis=1, channel_axis=2,
        #                                 fill_mode='constant', cval=0.)

        # if np.random.randint(0, 10) == 7:
        #     img_and_mask = flip_axis(img_and_mask, axis=1)

        # if np.random.randint(0, 10) == 7:
        #     img_and_mask = flip_axis(img_and_mask, axis=0)

        # if np.random.randint(0, 10) == 7:
        #     salt_pepper_noise(img_and_mask, salt=0.2, amount=0.04)

        if batch_of_images.ndim == 4:
            batch_of_images[i, ...] = img_and_mask[..., 0:img_and_mask.shape[2] // 2]
            batch_of_masks[i, ...] = img_and_mask[..., img_and_mask.shape[2] // 2:]
        if batch_of_images.ndim == 5:
            img_and_mask = img_and_mask.reshape(orig_shape)
            batch_of_images[i, ...] = img_and_mask[..., 0:img_and_mask.shape[2] // 2, :]
            batch_of_masks[i, ...] = img_and_mask[..., img_and_mask.shape[2] // 2:, :].round()

        # Ensure the masks did not get any non-binary values.
        # batch_of_masks = batch_of_masks.round()

    return (batch_of_images, batch_of_masks)


''' Make the generators threadsafe in case of multiple threads '''


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def generate_train_batches(root_path, train_list, net_input_shape, net, batch_size=1, shuff=True, aug_data=False):
    # Create placeholders for training
    img_batch = np.zeros((np.concatenate(((batch_size,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batch_size,), net_input_shape))), dtype=np.uint8)

    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan in enumerate(train_list):
            try:
                scan_name = scan[0]
                mask_name = scan[1]
                path_to_np = join(root_path, 'np_files', splitext(scan_name)[0] + '.npz')
                with np.load(path_to_np) as data:
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(splitext(scan_name)[0]))
                train_img, train_mask = convert_data_to_numpy(root_path, scan_name, net_input_shape,
                                                              mask_name=mask_name)
                if np.array_equal(train_img, np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            img_batch[count, :, :, 0] = train_img
            mask_batch[count, :, :, 0] = train_mask

            count += 1
            if count % batch_size == 0:
                count = 0
                if aug_data:
                    img_batch, mask_batch = augmentImages(img_batch, mask_batch)

                # mask_batch = mask_batch.astype(np.uint8)

                if debug:
                    plt.imshow(np.squeeze(img_batch[0, :, :, 0]), cmap='gray')
                    plt.imshow(np.squeeze(mask_batch[0, :, :, 0]), alpha=0.15)
                    plt.show()
                    plt.savefig(join(root_path, 'logs', scan_name), format='png', bbox_inches='tight')
                    plt.close()

                mask_batch_hot = to_one_hot(mask_batch, num_labels(root_path))
                # mask = mask_batch.squeeze(axis=-1)
                # mask_batch_hot = np.zeros((batch_size, train_mask.shape[0], train_mask.shape[1], 4))
                # mask_batch_hot[mask == 0, 0] = 1
                # mask_batch_hot[mask == 85, 1] = 1
                # mask_batch_hot[mask == 170, 2] = 1
                # mask_batch_hot[mask == 255, 3] = 1

                if net.find('caps') != -1:
                    img_masked = mask_batch_hot * img_batch
                    yield ([img_batch, mask_batch_hot],
                           [mask_batch_hot, img_masked[:, :, :, 0], img_masked[:, :, :, 1], img_masked[:, :, :, 2],
                            img_masked[:, :, :, 3]])
                    # yield (img_batch, mask_batch_hot)
                else:
                    yield (img_batch, mask_batch_hot)

        if count != 0:
            # if aug_data:
            #     img_batch[:count, ...], mask_batch[:count, ...] = augmentImages(img_batch[:count, ...],
            #                                                                     mask_batch[:count, ...])
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count, ...], mask_batch[:count, ...])


@threadsafe_generator
def generate_val_batches(root_path, val_list, net_input_shape, net, batch_size=1, shuff=1):
    # Create placeholders for validation
    img_batch = np.zeros((np.concatenate(((batch_size,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batch_size,), net_input_shape))), dtype=np.uint8)

    while True:
        if shuff:
            shuffle(val_list)
        count = 0
        for i, scan in enumerate(val_list):
            try:
                scan_name = scan[0]
                mask_name = scan[1]
                path_to_np = join(root_path, 'np_files', splitext(scan_name)[0] + 'npz')
                with np.load(path_to_np) as data:
                    val_img = data['img']
                    val_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(splitext(scan_name)[0]))
                val_img, val_mask = convert_data_to_numpy(root_path, scan_name, net_input_shape, mask_name=mask_name)
                if np.array_equal(val_img, np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            img_batch[count, :, :, 0] = val_img
            mask_batch[count, :, :, 0] = val_mask

            count += 1
            if count % batch_size == 0:
                count = 0

                mask_batch_hot = to_one_hot(mask_batch, num_labels(root_path))
                # mask = mask_batch.squeeze(axis=-1)
                # mask_batch_hot = np.zeros((batch_size, val_mask.shape[0], val_mask.shape[1], 4))
                # mask_batch_hot[mask == 0, 0] = 1
                # mask_batch_hot[mask == 85, 1] = 1
                # mask_batch_hot[mask == 170, 2] = 1
                # mask_batch_hot[mask == 255, 3] = 1
                if net.find('caps') != -1:
                    # yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch])
                    img_masked = mask_batch_hot * img_batch
                    yield ([img_batch, mask_batch_hot],
                           [mask_batch_hot, img_masked[:, :, :, 0], img_masked[:, :, :, 1], img_masked[:, :, :, 2],
                            img_masked[:, :, :, 3]])
                    # yield (img_batch, mask_batch_hot)
                else:
                    yield (img_batch, mask_batch_hot)

        if count != 0:
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count, ...], mask_batch[:count, ...])


@threadsafe_generator
def generate_test_batches(root_path, test_list, net_input_shape, batch_size=1):
    # Create placeholders for testing
    img_batch = np.zeros((np.concatenate(((batch_size,), net_input_shape))), dtype=np.float32)
    count = 0
    for i, scan_name in enumerate(test_list):
        try:
            scan_name = scan_name[0]
            path_to_np = join(root_path, 'np_files', basename(scan_name)[:-3] + 'npz')
            with np.load(path_to_np) as data:
                test_img = data['img']
        except:
            print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
            test_img = convert_data_to_numpy(root_path, scan_name, net_input_shape, no_masks=True)
            if np.array_equal(test_img, np.zeros(1)):
                continue
            else:
                print('\nFinished making npz file.')

        img_batch[count, :, :, 0] = test_img

        count += 1
        if count % batch_size == 0:
            count = 0
            yield (img_batch,)

    if count != 0:
        yield (img_batch[:count, :, :, :])


def num_labels(root_path):
    if any(x in root_path for x in ['Spectralis', 'Cirrus', 'Layers']):
        return 4
    else:
        return 6


def to_one_hot(values, n_labels):
    if values.ndim == 4:
        values = values.squeeze(axis=-1)
    return np.eye(n_labels)[values]

