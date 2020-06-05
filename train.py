'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for training models. Please see the README for details about training.
'''

from __future__ import print_function

import matplotlib
from math import ceil
from keras.callbacks import Callback

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()

from os.path import join
import numpy as np

from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf

from custom_losses import dice_hard, weighted_binary_crossentropy_loss, margin_loss, \
    WeightedCategoricalCrossEntropy, weighted_dice_coef, weighted_mse_loss, weighted_dice_loss, spread_loss, dice_soft, \
    weighted_spread_loss
from load_3D_data import load_class_weights, generate_train_batches, generate_val_batches

# CIRRUS_PIXEL_CLASS_WEIGHTS = {0: 0.003555870045612856, 1: 0.874566629820256, 2: 1.0, 3: 0.8723247732858718}
# CIRRUS_PRESENCE_CLASS_WEIGHTS = {0: 0.20247395833333331, 1: 0.6652406417112299, 2: 0.8885714285714286, 3: 1.0}
# SPECTRALIS_PIXEL_CLASS_WEIGHTS = {0: 0.0031737332265260555, 1: 0.5547157937848298, 2: 0.5589333800101778, 3: 1.0}
# SPECTRALIS_PRESENCE_CLASS_WEIGHTS = {0: 0.2568027210884354, 1: 0.6003976143141153, 2: 0.8908554572271387, 3: 1.0}
C_PIXEL = (0.003555870045612856, 0.874566629820256, 1.0, 0.8723247732858718)
C_PRESENCE = (0.20247395833333331, 0.6652406417112299, 0.8885714285714286, 1.0)
S_PIXEL = (0.0031737332265260555, 0.5547157937848298, 0.5589333800101778, 1.0)
S_PIXELS = (1., 0.00572137, 0.0056782,  0.00317373)
S_PRESENCE = (0.2568027210884354, 0.6003976143141153, 0.8908554572271387, 1.0)
S_PRESENCES = (1., 0.42772109, 0.28826531, 0.25680272)
S_PRESENCE_MSE = (1.0, 2.337972166998012, 3.4690265486725664, 3.8940397350993377)
S_PIXEL_MSE = (0.014573297188993577, 176.3305415840263, 177.67881191975138, 318.6781911942668)
CCE_WEIGHTS = (1.0, 174.78337156649349, 176.11227539183628, 315.08634425918416)


def get_loss(root, split, net, recon_wei, choice):
    if choice == 'w_bce':
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = weighted_binary_crossentropy_loss(pos_class_weight)
    elif choice == 'bce':
        loss = 'binary_crossentropy'
    elif choice == 'dice':
        loss = weighted_dice_loss(S_PRESENCE)
    elif choice == 'w_mar':
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=pos_class_weight)
    elif choice == 'mar':
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    elif choice == 'cce':
        loss = 'categorical_crossentropy'
    elif choice == 'scce':
        loss = 'sparse_categorical_crossentropy'
    elif choice == 'spread':
        loss = spread_loss(epoch_step=EpochCounter.counter)
    elif choice == 'w_spread':
        if 'Spectralis' in root:
            weights = np.array(S_PIXELS)
        else:
            weights = np.array(C_PRESENCE)
        loss = weighted_spread_loss(weights=weights, epoch_step=EpochCounter.counter)
    else:
        raise Exception("Unknow loss_type")

    if net.find('caps') != -1:
        return {'out_seg': loss, 'recon0': weighted_mse_loss(S_PIXEL_MSE[0]),
                'recon1': weighted_mse_loss(S_PIXEL_MSE[1]),
                'recon2': weighted_mse_loss(S_PIXEL_MSE[2]),
                'recon3': weighted_mse_loss(S_PIXEL_MSE[3])}, {'out_seg': 1., 'recon0': recon_wei,
                                                               'recon1': recon_wei,
                                                               'recon2': recon_wei,
                                                               'recon3': recon_wei}
        # return {'out_seg': loss}, None
    else:
        return loss, None


class EpochCounter(Callback):
    counter = K.variable(0.)

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        K.set_value(EpochCounter.counter, epoch + 1)


def get_callbacks(arguments):
    if arguments.net.find('caps') != -1:
        # monitor_name = 'val_out_seg_categorical_accuracy'
        monitor_name = 'val_out_seg_loss'
    else:
        # monitor_name = 'val_dice_hard'
        monitor_name = 'val_dice_soft'

    csv_logger = CSVLogger(join(arguments.log_dir, arguments.output_name + '_log_' + arguments.time + '.csv'),
                           separator=',')
    tb = TensorBoard(arguments.tf_log_dir, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(
        join(arguments.check_dir, arguments.output_name + '_model_' + arguments.time + '.hdf5'),
        monitor=monitor_name, save_best_only=True, save_weights_only=True,
        verbose=1, mode='max')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=5, verbose=1, mode='max')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=25, verbose=0, mode='max')

    epoch_counter = EpochCounter()

    return [model_checkpoint, csv_logger, lr_reducer, early_stopper, tb, epoch_counter]


def compile_model(args, net_input_shape, uncomp_model):
    # Set optimizer loss and metrics
    # if args.net == 'matwo':
    #     opt = Adadelta()
    # else:
    opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6)
    if args.net.find('caps') != -1:
        # metrics = {'out_seg': 'categorical_accuracy'}
        metrics = {'out_seg': weighted_dice_coef(S_PRESENCE)}
    elif args.net == 'matwo':
        # metrics = ['categorical_accuracy']
        metrics = [dice_soft]
    else:
        metrics = [dice_hard]

    loss, loss_weighting = get_loss(root=args.data_root_dir, split=args.split_num, net=args.net,
                                    recon_wei=args.recon_wei, choice=args.loss)

    # If using CPU or single GPU
    if args.gpus <= 1:
        uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
        return uncomp_model
    # If using multiple GPUs
    else:
        with tf.device("/cpu:0"):
            uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
            model = multi_gpu_model(uncomp_model, gpus=args.gpus)
            model.__setattr__('callback_model', uncomp_model)
        model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
        return model


def plot_training(training_history, arguments):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(arguments.net, fontsize=18)

    if arguments.net.find('caps') != -1:
        # ax1.plot(training_history.history['out_seg_categorical_accuracy'])
        # ax1.plot(training_history.history['val_out_seg_categorical_accuracy'])
        ax1.plot(training_history.history['out_seg_coef'])
        ax1.plot(training_history.history['val_out_seg_coef'])
    else:
        # ax1.plot(training_history.history['dice_hard'])
        # ax1.plot(training_history.history['val_dice_hard'])
        # ax1.plot(training_history.history['categorical_accuracy'])
        # ax1.plot(training_history.history['val_categorical_accuracy'])
        ax1.plot(training_history.history['dice_soft'])
        ax1.plot(training_history.history['val_dice_soft'])
    ax1.set_title('Dice Soft')
    ax1.set_ylabel('Dice', fontsize=12)
    ax1.legend(['Train', 'Val'], loc='upper left')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    if arguments.net.find('caps') != -1:
        # ax1.set_xticks(np.arange(0, len(training_history.history['out_seg_categorical_accuracy'])))
        ax1.set_xticks(np.arange(0, len(training_history.history['out_seg_coef'])))
    else:
        ax1.set_xticks(np.arange(0, len(training_history.history['dice_soft'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['loss'])
    ax2.plot(training_history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    ax1.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(join(arguments.output_dir, arguments.output_name + '_plots_' + arguments.time + '.png'))
    plt.close()


def train(args, train_list, val_list, u_model, net_input_shape):
    # Compile the loaded model
    model = compile_model(args=args, net_input_shape=net_input_shape, uncomp_model=u_model)
    # Set the callbacks
    callbacks = get_callbacks(args)

    # Training the network
    history = model.fit(
        generate_train_batches(args.data_root_dir, train_list, net_input_shape, net=args.net,
                               batch_size=args.batch_size, shuff=args.shuffle_data, aug_data=args.aug_data),
        max_queue_size=40, workers=4, use_multiprocessing=False,
        steps_per_epoch=ceil(len(train_list) / args.batch_size),
        validation_data=generate_val_batches(args.data_root_dir, val_list, net_input_shape, net=args.net,
                                             batch_size=args.batch_size, shuff=args.shuffle_data),
        validation_steps=ceil(len(val_list) / args.batch_size),  # Set validation stride larger to see more of the data.
        epochs=50,
        callbacks=callbacks,
        verbose=1)

    # Plot the training data collected
    plot_training(history, args)
