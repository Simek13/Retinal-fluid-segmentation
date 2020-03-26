'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the network definitions for the various capsule network architectures.
'''

from keras import layers, models
from keras import backend as K

K.set_image_data_format('channels_last')

from capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, PrimaryCaps2dMatwo, Caps2dMatwo, Mask, Length, \
    DualLength


def Matwo_CapsNet(input_shape, num_labels=6, is_training=True, routing_type='dual', routing=3,
                  data_format='channels_last'):
    padding = 'same'
    coord_add = True
    pos_dim = [4, 4]
    app_dim = [5, 5]
    level_caps = [5, 5, 6, 7]

    input_tensor = layers.Input(batch_shape=input_shape)

    x = PrimaryCaps2dMatwo(pos_dim=pos_dim, app_dim=app_dim, num_capsule=int(level_caps[0]), kernel_size=5,
                           strides=1, name='primary_caps', padding=padding, is_training=is_training,
                           data_format=data_format)(input_tensor)

    x = Caps2dMatwo(routings=1, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[1]), kernel_size=5, name='caps_1df_cd1', coord_add=coord_add,
                    padding=padding, strides=1, op="conv", is_training=is_training)(x)
    skip1 = x
    # 1/2
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[1]), kernel_size=5, name='caps_12_cd1', coord_add=coord_add,
                    padding=padding, strides=2, op="conv", is_training=is_training)(x)

    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[1]), kernel_size=5, name='caps_12_cd2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv", is_training=is_training)(x)
    skip2 = x

    # 1/4
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=5, name='caps_14_cd1', coord_add=coord_add,
                    padding=padding, strides=2, op="conv", is_training=is_training)(x)
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=5, name='caps_14_cd2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv", is_training=is_training)(x)
    skip3 = x

    # 1/8
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[3]), kernel_size=5, name='caps_18_cd1', coord_add=coord_add,
                    padding=padding, strides=2, op="conv", is_training=is_training)(x)
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[3]), kernel_size=5, name='caps_18_cd2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv", is_training=is_training)(x)
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[3]), kernel_size=5, name='caps_18_cd3', coord_add=coord_add,
                    padding=padding, strides=1, op="conv", is_training=is_training)(x)

    # 1/4
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[3]), kernel_size=4, name='caps_14_du1', coord_add=coord_add,
                    padding=padding, strides=2, op="deconv", is_training=is_training)(x)
    x = layers.Concatenate(axis=1, name='up_1')([x, skip3])
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[3]), kernel_size=5, name='caps_14_cu2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv", is_training=is_training)(x)

    # 1/2
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=4, name='caps_12_du1', coord_add=coord_add,
                    padding=padding, strides=2, op="deconv", is_training=is_training)(x)
    x = layers.Concatenate(axis=1, name='up_2')([x, skip2])
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=5, name='caps_12_cu2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv", is_training=is_training)(x)

    # 1
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=4, name='caps_1_du1', coord_add=coord_add,
                    padding=padding, strides=2, op="deconv", is_training=is_training)(x)
    x = layers.Concatenate(axis=1, name='up_3')([x, skip1])

    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=num_labels, kernel_size=1, name='caps_1_c2', coord_add=coord_add, padding=padding,
                    strides=1, op="conv", is_training=is_training)(x)

    prediction = DualLength(pos_dim=pos_dim, app_dim=app_dim)(x)

    model = models.Model(inputs=input_tensor, outputs=prediction)

    return model


def CapsNetR3(input_shape, n_class=4):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H, W, 1, C))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=3, name='conv_cap_2_1')(primary_caps)

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=3, name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=3, name='conv_cap_4_1')(conv_cap_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_1_1')(conv_cap_4_1)

    # Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=3, name='deconv_cap_1_2')(up_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=3, name='deconv_cap_2_2')(up_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)

    # Skip connection
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps')(up_3)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1] + (4,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)

    def shared_decoder(mask_layer):
        # recon_remove_dim = layers.Reshape((H, W, A))(mask_layer)

        mask0 = layers.Lambda(lambda x: x[:, :, :, 0, :])(mask_layer)
        mask1 = layers.Lambda(lambda x: x[:, :, :, 1, :])(mask_layer)
        mask2 = layers.Lambda(lambda x: x[:, :, :, 2, :])(mask_layer)
        mask3 = layers.Lambda(lambda x: x[:, :, :, 3, :])(mask_layer)

        recon_01 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                 activation='relu', name='recon_01')(mask0)
        recon_11 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                 activation='relu', name='recon_11')(mask1)
        recon_21 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                 activation='relu', name='recon_21')(mask2)
        recon_31 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                 activation='relu', name='recon_31')(mask3)

        recon_02 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                 activation='relu', name='recon_02')(recon_01)
        recon_12 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                 activation='relu', name='recon_12')(recon_11)
        recon_22 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                 activation='relu', name='recon_22')(recon_21)
        recon_32 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                 activation='relu', name='recon_32')(recon_31)

        out_recon0 = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                   activation='sigmoid', name='out_recon0')(recon_02)
        out_recon1 = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                   activation='sigmoid', name='out_recon1')(recon_12)
        out_recon2 = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                   activation='sigmoid', name='out_recon2')(recon_22)
        out_recon3 = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                   activation='sigmoid', name='out_recon3')(recon_32)

        out_recon = layers.Concatenate(axis=-1, name='out_recon')([out_recon0, out_recon1, out_recon2, out_recon3])

        return out_recon

    # masked_by_y = Mask()([shared_decoder(seg_caps), y])  # The true label is used to mask the output of capsule layer. For training
    # masked = Mask()(shared_decoder(seg_caps))  # Mask using the capsule with maximal length. For prediction
    masked_by_y_dec = shared_decoder(masked_by_y)

    masked_by_y_dec0 = layers.Lambda(lambda x: x[:, :, :, 0], name='recon0')(masked_by_y_dec)
    masked_by_y_dec1 = layers.Lambda(lambda x: x[:, :, :, 1], name='recon1')(masked_by_y_dec)
    masked_by_y_dec2 = layers.Lambda(lambda x: x[:, :, :, 2], name='recon2')(masked_by_y_dec)
    masked_by_y_dec3 = layers.Lambda(lambda x: x[:, :, :, 3], name='recon3')(masked_by_y_dec)

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, masked_by_y_dec0, masked_by_y_dec1, masked_by_y_dec2,
                                                       masked_by_y_dec3])
    # train_model = models.Model(inputs=x, outputs=out_seg)
    # eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])
    eval_model = models.Model(inputs=x, outputs=out_seg)

    # manipulate model
    # noise = layers.Input(shape=((H, W, C, A)))
    # noised_seg_caps = layers.Add()([seg_caps, noise])
    # masked_noised_y = Mask()([noised_seg_caps, y])
    # manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model


def CapsNetR1(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=1, name='conv_cap_2_1')(primary_caps)

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=3, name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=3, name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='conv_cap_4_1')(conv_cap_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_1_1')(conv_cap_4_1)

    # Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=1, name='deconv_cap_1_2')(up_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=1, name='deconv_cap_2_2')(up_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)

    # Skip connection
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=1, name='seg_caps')(up_3)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1] + (1,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=((H.value, W.value, C.value, A.value)))
    noised_seg_caps = layers.Add()([seg_caps, noise])
    masked_noised_y = Mask()([noised_seg_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model


def CapsNetBasic(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                routings=3, name='seg_caps')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()
    y = layers.Input(shape=input_shape[:-1] + (1,))
    masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=((H.value, W.value, C.value, A.value)))
    noised_seg_caps = layers.Add()([seg_caps, noise])
    masked_noised_y = Mask()([noised_seg_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model
