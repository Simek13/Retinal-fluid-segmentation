'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for testing models. Please see the README for details about testing.
'''

from __future__ import print_function

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()

from os.path import join
from os import makedirs
import csv
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters
from metrics import assd, DiceMetric, JaccardMetric
from load_3D_data import num_labels
from scipy.special import softmax
from keras import backend as K

K.set_image_data_format('channels_last')
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc

from load_3D_data import generate_test_batches


def threshold_mask(raw_output, threshold):
    if threshold == 0:
        try:
            threshold = filters.threshold_otsu(raw_output)
        except:
            threshold = 0.5

    print('\tThreshold: {}'.format(threshold))

    raw_output[raw_output > threshold] = 1
    raw_output[raw_output < 1] = 0

    all_labels = measure.label(raw_output)
    props = measure.regionprops(all_labels)
    props.sort(key=lambda x: x.area, reverse=True)
    thresholded_mask = np.zeros(raw_output.shape)

    if len(props) >= 2:
        if props[0].area / props[1].area > 5:  # if the largest is way larger than the second largest
            thresholded_mask[all_labels == props[0].label] = 1  # only turn on the largest component
        else:
            thresholded_mask[all_labels == props[0].label] = 1  # turn on two largest components
            thresholded_mask[all_labels == props[1].label] = 1
    elif len(props):
        thresholded_mask[all_labels == props[0].label] = 1

    thresholded_mask = scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)

    return thresholded_mask


def test(args, test_list, model_list, net_input_shape):
    if args.weights_path == '':
        weights_path = join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5')
    else:
        weights_path = join(args.data_root_dir, args.weights_path)

    output_dir = join(args.data_root_dir, 'results', args.net)
    raw_out_dir = join(output_dir, 'raw_output')
    fin_out_dir = join(output_dir, 'final_output')
    fig_out_dir = join(output_dir, 'qual_figs')
    try:
        makedirs(raw_out_dir)
    except:
        pass
    try:
        makedirs(fin_out_dir)
    except:
        pass
    try:
        makedirs(fig_out_dir)
    except:
        pass

    if len(model_list) > 1:
        eval_model = model_list[1]
    else:
        eval_model = model_list[0]
    try:
        eval_model.load_weights(weights_path)
    except:
        print('Unable to find weights path. Testing with random weights.')
    # print_summary(model=eval_model, positions=[.38, .65, .75, 1.])
    eval_model.summary()
    # Set up placeholders
    outfile = ''
    dice_arr = {}
    outfile += 'dice_'
    jacc_arr = {}
    outfile += 'jacc_'

    # Testing the network
    print('Testing... This will take some time...')

    with open(join(output_dir, args.save_prefix + outfile + 'scores.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        row = ['Scan Name', 'Dice Coefficient', 'Jaccard Index']

        writer.writerow(row)

        num_l = num_labels(args.data_root_dir)
        label_stats = {i: [] for i in range(num_l)}
        prediction_scores = {i: [] for i in range(num_l)}
        # labels = []
        # prediction_scores = []

        for scan in tqdm(test_list):
            volume = scan[0]
            if not volume in dice_arr:
                dice_arr[volume] = []
                jacc_arr[volume] = []
            img = Image.open(join(args.data_root_dir, 'test', volume, 'images', scan[1]))
            resized_img = img.resize((net_input_shape[1], net_input_shape[0]))

            output_array = eval_model.predict(generate_test_batches(args.data_root_dir, [scan],
                                                                    net_input_shape,
                                                                    batch_size=args.batch_size),
                                              steps=1, max_queue_size=1, workers=1,
                                              use_multiprocessing=False, verbose=1)

            if args.net.find('caps') != -1:
                output = output_array
                # recon = output_array[1]
            elif args.net == 'matwo':
                output = output_array
            else:
                output = output_array[:, :, :, 0]


            print('Segmenting Output')
            shade_number = output.shape[-1]
            shades = np.linspace(0, 255, shade_number)
            output = np.argmax(output, axis=-1)
            output = np.squeeze(output, axis=0)
            output_bin = output.copy()
            for j, shade in enumerate(shades):
                output_bin[output_bin == j] = int(shade)
            output_bin = output_bin.astype('uint8')
            output_mask = sitk.GetImageFromArray(output_bin)

            print('Saving Output')
            sitk.WriteImage(output_mask, join(fin_out_dir, scan[1][:-4] + '_final_output' + scan[1][-4:]))

            mask_data = np.array(
                Image.open(join(args.data_root_dir, 'test', volume, 'masks', scan[2])).resize((net_input_shape[1], net_input_shape[0]), Image.NEAREST),
                dtype=np.uint8)

            output_array = softmax(output_array, axis=-1)
            for j in range(num_l):
                y = np.zeros(mask_data.shape)
                y[mask_data == j] = 1
                label_stats[j].append(y.flatten())
                score = output_array[..., j]
                prediction_scores[j].append(score.flatten())

            # labels.append(mask_data.flatten())
            # prediction_scores.append(softmax(output_array, axis=-1).reshape(-1, output_array.shape[-1]))

            # Plot Qual Figure
            print('Creating Qualitative Figure for Quick Reference')
            if 'Cirrus' in args.data_root_dir:
                f, ax = plt.subplots(1, 3, figsize=(10, 5))
            else:
                f, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(img, alpha=1, cmap='gray')
            ax[0].set_title('Originalni sken')
            ax[1].imshow(resized_img, alpha=1, cmap='gray')
            ax[1].imshow(mask_to_rgb(output), alpha=0.5)
            # ax[1].imshow(output, alpha=1, cmap='gray')
            ax[1].set_title('Izlaz iz mreže')
            ax[2].imshow(resized_img, alpha=1, cmap='gray')
            ax[2].imshow(mask_to_rgb(mask_data), alpha=0.5)
            # ax[2].imshow(mask_data, alpha=1, cmap='gray')
            ax[2].set_title('Referentna maska')
            # ax[0].imshow(mask_data, alpha=1, cmap='gray')
            # ax[1].imshow(mask_to_rgb(mask_data), alpha=1)

            fig = plt.gcf()
            # fig.suptitle(scan[1][:-4])

            plt.savefig(join(fig_out_dir, scan[1][:-4] + '_qual_fig' + '.png'),
                        format='png', bbox_inches='tight')
            plt.close('all')

            row = [scan[1][:-4]]
            print('Computing Dice')
            dice = DiceMetric()
            dice_score = dice(output, mask_data, range(num_l))
            dice_arr[volume].append(dice_score)
            print('\tDice: {}'.format(dice_score))
            row.append(dice_score)
            print('Computing Jaccard')
            jaccard = JaccardMetric()
            jacc_score = jaccard(output, mask_data, range(num_l))
            jacc_arr[volume].append(jacc_score)
            print('\tJaccard: {}'.format(jacc_score))
            row.append(jacc_score)

            writer.writerow(row)

        writer.writerow('Volume averages:')
        volume_dice_avgs = []
        volume_jacc_avgs = []
        for v in dice_arr:
            volume_dice_avg = np.mean(np.stack(dice_arr[v]), axis=0)
            volume_jacc_avg = np.mean(np.stack(jacc_arr[v]), axis=0)
            volume_dice_avgs.append(volume_dice_avg)
            volume_jacc_avgs.append(volume_jacc_avg)
            row = [v, volume_dice_avg, volume_jacc_avg]
            writer.writerow(row)
        volume_dice_avgs = np.stack(volume_dice_avgs)
        dice_mean = np.mean(volume_dice_avgs, axis=0)
        volume_jacc_avgs = np.stack(volume_jacc_avgs)
        jacc_mean = np.mean(volume_jacc_avgs, axis=0)
        row = ['Average Scores', dice_mean, jacc_mean]
        writer.writerow(row)

        fpr = []
        tpr = []
        auc_scores = []
        for i in range(1, num_l):
            labels = np.stack(label_stats[i]).flatten()
            scores = np.stack(prediction_scores[i]).flatten()
            fpr_rf, tpr_rf, thresholds_rf = roc_curve(labels, scores)
            fpr.append(fpr_rf)
            tpr.append(tpr_rf)
            auc_rf = auc(fpr_rf, tpr_rf)
            auc_scores.append(auc_rf)
        plot_roc_curves(args, fpr, tpr, auc_scores, output_dir)

        # labels = np.stack(labels).flatten()
        # prediction_scores = np.stack(prediction_scores).reshape(-1, output_array.shape[-1])
        #
        # roc_auc = roc_auc_score(labels, prediction_scores, multi_class='ovr')
        print('AUC score: {}'.format(auc_scores))
        row = ['AUC Scores', auc_scores]
        writer.writerow(row)

    print('Done.')


def plot_roc_curves(args, fpr, tpr, auc_scores, output_dir):
    plt.figure()
    lw = 2
    colors = ['red', 'green', 'blue']
    fluids = ['IRF', 'SRF', 'PED']
    for i in range(len(auc_scores)):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw, label=fluids[i] + ' ROC curve (area = %0.2f)' % auc_scores[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if 'Cirrus' in args.data_root_dir:
        plt.title('Cirrus')
    else:
        plt.title('Spectralis')
    plt.legend(loc="lower right")
    plt.savefig(join(output_dir, 'roc_curves.png'),
                format='png', bbox_inches='tight')
    plt.close('all')


def mask_to_rgb(mask):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask == 1, 0] = 255
    rgb_mask[mask == 2, 1] = 255
    rgb_mask[mask == 3, 2] = 255
    return rgb_mask


def to_shades(mask):
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    new_mask[mask == 1] = 85
    new_mask[mask == 2] = 170
    new_mask[mask == 3] = 255

