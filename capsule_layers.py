'''
Capsules for Object Segmentation (SegCaps)
Original Paper: https://arxiv.org/abs/1804.04241
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the definitions of the various capsule layers and dynamic routing and squashing functions.
'''

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import initializers, layers, activations as actv
from keras.utils.conv_utils import conv_output_length, deconv_length
from tensorflow.keras.activations import softmax
import numpy as np


class Length(layers.Layer):
    def __init__(self, num_classes, seg=True, **kwargs):
        super(Length, self).__init__(**kwargs)
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        self.seg = seg

    def call(self, inputs, **kwargs):
        return softmax(tf.norm(inputs, axis=-1), axis=-1)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 5:
            input_shape = input_shape[0:-2] + input_shape[-1:]
        if self.seg:
            return input_shape[:-1] + (self.num_classes,)
        else:
            return input_shape[:-1]

    def get_config(self):
        config = {'num_classes': self.num_classes, 'seg': self.seg}
        base_config = super(Length, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DualLength(layers.Layer):
    def __init__(self, pos_dim, app_dim, **kwargs):
        super(DualLength, self).__init__(**kwargs)
        self.pos_dim = pos_dim
        self.app_dim = app_dim

    def call(self, inputs, **kwargs):
        x = tf.transpose(inputs, (0, 3, 4, 1, 2))
        x_mat, x_mat2 = tf.split(x, [np.product(self.pos_dim), np.product(self.app_dim)], axis=-1)
        pred = tf.norm(x_mat, axis=-1) * tf.norm(x_mat2, axis=-1)
        return softmax(pred)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[3:] + (input_shape[1],)

    def get_config(self):
        config = {'pos_dim': self.pos_dim, 'app_dim': self.app_dim}
        base_config = super(DualLength, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Mask(layers.Layer):
    def __init__(self, resize_masks=False, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.resize_masks = resize_masks

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            input, mask = inputs
            _, hei, wid, _, _ = input.get_shape()
            if self.resize_masks:
                mask = tf.image.resize_bicubic(mask, (hei, wid))
            mask = K.expand_dims(mask, -1)
            if input.get_shape().ndims == 3:
                masked = K.batch_flatten(mask * input)
            else:
                masked = mask * input

        else:
            if inputs.get_shape().ndims == 3:
                x = K.sqrt(K.sum(K.square(inputs), -1))
                mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
                masked = K.batch_flatten(K.expand_dims(mask, -1) * inputs)
            else:
                masked = inputs

        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            if len(input_shape[0]) == 3:
                return tuple([None, input_shape[0][1] * input_shape[0][2]])
            else:
                return input_shape[0]
        else:  # no true label provided
            if len(input_shape) == 3:
                return tuple([None, input_shape[1] * input_shape[2]])
            else:
                return input_shape

    def get_config(self):
        config = {'resize_masks': self.resize_masks}
        base_config = super(Mask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(ConvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                        self.input_num_atoms, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):
        input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
        input_shape = K.shape(input_transposed)
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[0] * input_shape[1], self.input_height, self.input_width, self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))

        conv = K.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
                        padding=self.padding, data_format='channels_last')

        votes_shape = K.shape(conv)
        _, conv_height, conv_width, _ = conv.get_shape()

        votes = K.reshape(conv, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
                                 self.num_capsule, self.num_atoms])
        votes.set_shape((None, self.input_num_capsule, conv_height, conv_width,
                         self.num_capsule, self.num_atoms))

        logit_shape = K.stack([
            input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = K.tile(self.b, [conv_height, conv_width, 1, 1])

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        return activations

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size,
                padding=self.padding,
                stride=self.strides,
                dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'strides': self.strides,
            'padding': self.padding,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeconvCapsuleLayer(layers.Layer):
    def __init__(self, kernel_size, num_capsule, num_atoms, scaling=2, upsamp_type='deconv', padding='same', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(DeconvCapsuleLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.scaling = scaling
        self.upsamp_type = upsamp_type
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        if self.upsamp_type == 'subpix':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.input_num_atoms,
                                            self.num_capsule * self.num_atoms * self.scaling * self.scaling],
                                     initializer=self.kernel_initializer,
                                     name='W')
        elif self.upsamp_type == 'resize':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.input_num_atoms, self.num_capsule * self.num_atoms],
                                     initializer=self.kernel_initializer, name='W')
        elif self.upsamp_type == 'deconv':
            self.W = self.add_weight(shape=[self.kernel_size, self.kernel_size,
                                            self.num_capsule * self.num_atoms, self.input_num_atoms],
                                     initializer=self.kernel_initializer, name='W')
        else:
            raise NotImplementedError('Upsampling must be one of: "deconv", "resize", or "subpix"')

        self.b = self.add_weight(shape=[1, 1, self.num_capsule, self.num_atoms],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):
        input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
        input_shape = K.shape(input_transposed)
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[1] * input_shape[0], self.input_height, self.input_width, self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))

        if self.upsamp_type == 'resize':
            upsamp = K.resize_images(input_tensor_reshaped, self.scaling, self.scaling, 'channels_last')
            outputs = K.conv2d(upsamp, kernel=self.W, strides=(1, 1), padding=self.padding, data_format='channels_last')
        elif self.upsamp_type == 'subpix':
            conv = K.conv2d(input_tensor_reshaped, kernel=self.W, strides=(1, 1), padding='same',
                            data_format='channels_last')
            outputs = tf.depth_to_space(conv, self.scaling)
        else:
            batch_size = input_shape[1] * input_shape[0]

            # Infer the dynamic output shape:
            out_height = deconv_length(self.input_height, self.scaling, self.kernel_size, self.padding,
                                       output_padding=None)
            out_width = deconv_length(self.input_width, self.scaling, self.kernel_size, self.padding,
                                      output_padding=None)
            output_shape = (batch_size, out_height, out_width, self.num_capsule * self.num_atoms)

            outputs = K.conv2d_transpose(input_tensor_reshaped, self.W, output_shape, (self.scaling, self.scaling),
                                         padding=self.padding, data_format='channels_last')

        votes_shape = K.shape(outputs)
        _, conv_height, conv_width, _ = outputs.get_shape()

        votes = K.reshape(outputs, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
                                    self.num_capsule, self.num_atoms])
        votes.set_shape((None, self.input_num_capsule, conv_height, conv_width,
                         self.num_capsule, self.num_atoms))

        logit_shape = K.stack([
            input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = K.tile(self.b, [votes_shape[1], votes_shape[2], 1, 1])

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        return activations

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        output_shape[1] = deconv_length(output_shape[1], self.scaling, self.kernel_size, self.padding,
                                        output_padding=None)
        output_shape[2] = deconv_length(output_shape[2], self.scaling, self.kernel_size, self.padding,
                                        output_padding=None)
        output_shape[3] = self.num_capsule
        output_shape[4] = self.num_atoms

        return tuple(output_shape)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'scaling': self.scaling,
            'padding': self.padding,
            'upsamp_type': self.upsamp_type,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(DeconvCapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PrimaryCaps2dMatwo(layers.Layer):
    def __init__(self, kernel_size, num_capsule, pos_dim, app_dim, strides=1, padding='same',
                 kernel_initializer='truncated_normal', activation='relu', data_format="channels_last",
                 is_training=False, debug_print=True, **kwargs):
        super(PrimaryCaps2dMatwo, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.pos_dim = pos_dim
        self.app_dim = app_dim
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activation
        self.data_format = data_format
        self.is_training = is_training
        self.debug_print = debug_print

    def build(self, input_shape):
        assert len(input_shape) == 4, "The input Tensor should have shape=[batch/N, height/H_0, width/W_0, channel/z_0]"
        self.batch = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.channels = input_shape[3]

        self.W = self.add_weight(
            shape=[self.kernel_size, self.kernel_size, self.channels,
                   self.num_capsule * self.app_dim[0] * self.app_dim[1]],
            initializer=self.kernel_initializer,
            name='W')

        self.W_app = self.add_weight(shape=[1, 1, 1, self.num_capsule * self.app_dim[1] * self.app_dim[1]],
                                     initializer=self.kernel_initializer,
                                     name='W_app')

        self.built = True

    def call(self, input_tensor, training=None):
        # Create the appearance projection matrix
        ones_kernel = K.ones([1, 1, 1, 1])
        # mult_app_all = layers.Conv2D(self.num_capsule * self.app_dim[1] * self.app_dim[1], kernel_size=1, strides=1,
        #                              kernel_initializer=self.kernel_initializer, padding=self.padding,
        #                              use_bias=False, trainable=self.is_training, name='mult_app')(ones_kernel)
        mult_app_all = K.conv2d(ones_kernel, self.W_app, (1, 1),
                                padding=self.padding, data_format=self.data_format)
        mult_app = K.reshape(mult_app_all, [self.num_capsule, self.app_dim[1], self.app_dim[1]])

        # Extract appearance value
        # u_t_app = layers.Conv2D(self.num_capsule * self.app_dim[0] * self.app_dim[
        #     1], kernel_size=self.kernel_size, strides=self.strides,
        #                         kernel_initializer=self.kernel_initializer,
        #                         activation=self.activation, padding=self.padding, use_bias=True,
        #                         data_format=self.data_format, trainable=self.is_training,
        #                         name='spatial_k')(input_tensor)
        u_t_app = K.conv2d(input_tensor, self.W, (self.strides, self.strides),
                           padding=self.padding, data_format=self.data_format)

        H_1 = u_t_app.get_shape()[1]
        W_1 = u_t_app.get_shape()[2]

        # Initialize the pose matrix with identity
        u_t_pos = K.zeros([self.batch, H_1, W_1, self.num_capsule, self.pos_dim[0], self.pos_dim[1]], dtype='float32')
        u_t_pos = K.reshape(u_t_pos, [self.batch * H_1 * W_1, self.num_capsule, self.pos_dim[0], self.pos_dim[1]])
        identity = tf.Variable(lambda: K.eye(self.pos_dim[1]))
        identity = identity[self.pos_dim[1] - self.pos_dim[0]:, :]
        u_t_pos += identity
        u_hat_t_pos = K.reshape(u_t_pos, [self.batch, H_1, W_1, self.num_capsule, np.prod(self.pos_dim)])

        # Apply the matrix multiplication to the appearance matrix
        u_t_app = K.reshape(u_t_app, [self.batch, H_1, W_1, self.num_capsule, self.app_dim[0], self.app_dim[1]])
        u_t_app = matmult2d(u_t_app, mult_app)
        u_hat_t_app = K.reshape(u_t_app, [self.batch, H_1, W_1, self.num_capsule, np.prod(self.app_dim)])

        # Squash the appearance matrix (Psquashing the pose won't change it)
        v_pos = u_hat_t_pos
        v_app = matwo_squash(u_hat_t_app)

        v = K.concatenate([v_pos, v_app], axis=-1)
        outputs = tf.transpose(v, (
            0, 3, 4, 1,
            2))  # [t_1, N, H_1, W_1, z_1] => [N, t, z , H, W] #[N, H_1, W_1, t_1, z_1] => [N, t, z , H_1, W_1]

        if self.debug_print:
            print_primary_matwocaps_parameters(inputs=input_tensor,
                                               outputs=outputs,
                                               capsule_types=self.num_capsule,
                                               app_dim=self.app_dim,
                                               pos_dim=self.pos_dim,
                                               kernel_size=self.kernel_size,
                                               name=self.name,
                                               is_training=self.trainable,
                                               padding=self.padding,
                                               strides=self.strides)

        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:3]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size,
                padding=self.padding,
                stride=self.strides,
                dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + (self.num_capsule, np.product(self.app_dim) + np.product(self.pos_dim)) + \
               tuple(new_space)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'pos_dim': self.pos_dim,
            'app_dim': self.app_dim,
            'strides': self.strides,
            'padding': self.padding,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': self.activation,
            'data_format': self.data_format,
            'is_training': self.is_training,
            'debug_print': self.debug_print
        }
        base_config = super(PrimaryCaps2dMatwo, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Caps2dMatwo(layers.Layer):
    def __init__(self, kernel_size, num_capsule, pos_dim, app_dim, routing_type, routings, strides, op='conv',
                 padding='same',
                 kernel_initializer='truncated_normal', coord_add=True, is_training=False, debug_print=True, **kwargs):
        super(Caps2dMatwo, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.pos_dim = pos_dim
        self.app_dim = app_dim
        self.op = op
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.routing_type = routing_type
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.coord_add = coord_add
        self.is_training = is_training
        self.debug_print = debug_print

    def build(self, input_shape):
        assert len(
            input_shape) == 5, "The input Tensor should have shape=[batch/N, capsule_type/t_0, channel/z_0, height/H_0, width/W_0]"
        self.batch = input_shape[0]
        self.input_num_capsule = input_shape[1]
        self.channels = input_shape[2]
        self.input_height = input_shape[3]
        self.input_width = input_shape[4]

        self.W = []
        if self.op == 'conv':
            shape = [self.kernel_size, self.kernel_size, 1,
                     self.num_capsule]
        else:
            shape = [self.kernel_size, self.kernel_size,
                     self.num_capsule, 1]
        for i in range(self.input_num_capsule):
            self.W.append(self.add_weight(
                shape=shape,
                initializer=self.kernel_initializer,
                name='W_{}'.format(i)))

        self.W_app = self.add_weight(
            shape=[1, 1, 1, self.input_num_capsule * self.num_capsule * self.app_dim[1] * self.app_dim[1]],
            initializer=self.kernel_initializer,
            name='W_app')

        self.W_pos = self.add_weight(
            shape=[1, 1, 1, self.input_num_capsule * self.num_capsule * self.pos_dim[1] * self.pos_dim[1]],
            initializer=self.kernel_initializer,
            name='W_pos')

        self.b_app = self.add_weight(shape=[1, 1, 1, self.input_num_capsule * self.num_capsule],
                                     initializer=self.kernel_initializer,
                                     name='b_app')

        self.built = True

    def call(self, input_tensor, training=None):
        z_app = np.product(self.app_dim)
        z_pos = np.product(self.pos_dim)

        # Initialize weight
        # Learning transformation matrices for pose and appearance and appearance bias
        # There are t0*t1 transformation matrices
        ones_kernel = K.ones([1, 1, 1, 1])

        mult_pos_all = K.conv2d(ones_kernel, self.W_pos, (1, 1),
                                padding=self.padding, data_format='channels_last')
        mult_pos_all = K.reshape(mult_pos_all,
                                 [self.input_num_capsule, self.pos_dim[1] * self.pos_dim[1], self.num_capsule])

        mult_app_all = K.conv2d(ones_kernel, self.W_app, (1, 1),
                                padding=self.padding, data_format='channels_last')
        mult_app_all = K.reshape(mult_app_all,
                                 [self.input_num_capsule, self.app_dim[1] * self.app_dim[1], self.num_capsule])

        bias_app = K.conv2d(ones_kernel, self.b_app, (1, 1),
                            padding=self.padding, data_format='channels_last')
        bias_app = K.reshape(bias_app, [self.input_num_capsule, self.num_capsule])

        # For each child (input) capsule (t_0) project into all parent (output) capsule domain (t_1)
        idx_all = 0
        u_hat_t_list = []

        # Split input_tensor by capsule types
        u_t_list = [K.squeeze(u_t, axis=1) for u_t in tf.split(input_tensor, self.input_num_capsule, axis=1)]
        for u_t in u_t_list:
            u_t = K.reshape(u_t, [self.batch * self.channels, self.input_height, self.input_width, 1])
            u_t.set_shape((None, self.input_height, self.input_width, 1))

            # Apply spatial kernel
            # Incorporating local neighborhood information by learning a convolution kernel of size k x k for the pose
            # and appearance matrices Pi and Ai
            if self.op == "conv":
                u_spat_t = K.conv2d(u_t, self.W[idx_all], (self.strides, self.strides),
                                    padding=self.padding, data_format='channels_last')
            elif self.op == "deconv":
                out_height = deconv_length(self.input_height, self.strides, self.kernel_size, self.padding,
                                           output_padding=None)
                out_width = deconv_length(self.input_width, self.strides, self.kernel_size, self.padding,
                                          output_padding=None)
                output_shape = (self.batch * self.channels, out_height, out_width, self.num_capsule)
                u_spat_t = K.conv2d_transpose(u_t, self.W[idx_all], output_shape, (self.strides, self.strides),
                                              padding=self.padding, data_format='channels_last')
            else:
                raise ValueError("Wrong type of operation for capsule")
            # Some shape operation
            H_1 = u_spat_t.get_shape()[1]
            W_1 = u_spat_t.get_shape()[2]
            u_spat_t = K.reshape(u_spat_t, [self.batch, self.channels, H_1, W_1, self.num_capsule])
            u_spat_t = tf.transpose(u_spat_t, (0, 2, 3, 4, 1))
            u_spat_t = K.reshape(u_spat_t, [self.batch, H_1, W_1, self.num_capsule * self.channels])

            # Split convolution output of input_tensor to Pose and Appearance matrices
            u_t_pos, u_t_app = tf.split(u_spat_t, [self.num_capsule * z_pos, self.num_capsule * z_app], axis=-1)
            u_t_pos = K.reshape(u_t_pos, [self.batch, H_1, W_1, self.num_capsule, self.pos_dim[0], self.pos_dim[1]])
            u_t_app = K.reshape(u_t_app, [self.batch, H_1, W_1, self.num_capsule, self.app_dim[0], self.app_dim[1]])

            # Gather projection matrices and bias
            # Take appropriate capsule type
            mult_pos = tf.gather(mult_pos_all, idx_all, axis=0)
            mult_pos = K.reshape(mult_pos, [self.num_capsule, self.pos_dim[1], self.pos_dim[1]])
            mult_app = tf.gather(mult_app_all, idx_all, axis=0)
            mult_app = K.reshape(mult_app, [self.num_capsule, self.app_dim[1], self.app_dim[1]])
            bias = tf.reshape(tf.gather(bias_app, idx_all, axis=0), (1, 1, 1, self.num_capsule, 1, 1))

            u_t_app += bias

            # Prepare the pose projection matrix

            # Why does he normalize second dimension?
            mult_pos = K.l2_normalize(mult_pos, axis=-2)
            if self.coord_add:
                # Zašto dodaje na prvu dimenziju normalizirane koordinate?
                mult_pos = coordinate_addition(mult_pos,
                                               [1, H_1, W_1, self.num_capsule, self.pos_dim[1], self.pos_dim[1]])

            u_t_pos = matmult2d(u_t_pos, mult_pos)
            u_t_app = matmult2d(u_t_app, mult_app)

            # Store the result
            u_hat_t_pos = K.reshape(u_t_pos, [self.batch, H_1, W_1, self.num_capsule, z_pos])
            u_hat_t_app = K.reshape(u_t_app, [self.batch, H_1, W_1, self.num_capsule, z_app])
            u_hat_t = K.concatenate([u_hat_t_pos, u_hat_t_app], axis=-1)
            u_hat_t_list.append(u_hat_t)

            idx_all += 1

        u_hat_t_list = K.stack(u_hat_t_list, axis=-2)
        u_hat_t_list = tf.transpose(u_hat_t_list,
                                    [0, 5, 1, 2, 4, 3])  # [N, H, W, t_1, t_0, z]  = > [N, z, H_1, W_1, t_0, t_1]

        # Routing operation
        if self.routings > 0:
            if self.routing_type is 'dynamic':
                if type(self.routings) is list:
                    self.routings = self.routings[-1]
                c_t_list = routing2d(routing=self.routings, t_0=self.input_num_capsule,
                                     u_hat_t_list=u_hat_t_list)  # [T1][N,H,W,to]
            elif self.routing_type is 'dual':
                if type(self.routings) is list:
                    self.routings = self.routings[-1]
                c_t_list = dual_routing2d(routing=self.routings, t_0=self.input_num_capsule, u_hat_t_list=u_hat_t_list,
                                          z_app=z_app,
                                          z_pos=z_pos)  # [T1][N,H,W,to]
            else:
                raise ValueError(self.routing_type + ' is an invalid routing; try dynamic or dual')
        else:
            self.routing_type = 'NONE'
            c = K.ones([self.batch, H_1, W_1, self.input_num_capsule, self.num_capsule])
            c_t_list = [K.squeeze(c_t, axis=-1) for c_t in tf.split(c, self.num_capsule, axis=-1)]

        # Form each parent capsule through the weighted sum of all child capsules
        r_t_mul_u_hat_t_list = []
        u_hat_t_list_ = [(K.squeeze(u_hat_t, axis=-1)) for u_hat_t in tf.split(u_hat_t_list, self.num_capsule, axis=-1)]
        for c_t, u_hat_t in zip(c_t_list, u_hat_t_list_):
            r_t = K.expand_dims(c_t, axis=1)
            r_t_mul_u_hat_t_list.append(K.sum(r_t * u_hat_t, axis=-1))

        p = r_t_mul_u_hat_t_list
        p = K.stack(p, axis=1)
        p_pos, p_app = tf.split(p, [z_pos, z_app], axis=2)

        # Squash the weighted sum to form the final parent capsule
        v_pos = Psquash(p_pos, axis=2)
        v_app = matwo_squash(p_app, axis=2)

        outputs = K.concatenate([v_pos, v_app], axis=2)

        if self.debug_print:
            print_matwocaps_parameters(inputs=input_tensor,
                                       outputs=outputs,
                                       routing=self.routings,
                                       coord_add=self.coord_add,
                                       routing_type=self.routing_type,
                                       capsule_types=self.num_capsule,
                                       app_dim=self.app_dim,
                                       pos_dim=self.pos_dim,
                                       kernel_size=self.kernel_size,
                                       name=self.name,
                                       is_training=self.is_training,
                                       padding=self.padding,
                                       strides=self.strides)

        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[3:]
        if self.op == "conv":
            new_space = []
            for i in range(len(space)):
                new_dim = conv_output_length(
                    space[i],
                    self.kernel_size,
                    padding=self.padding,
                    stride=self.strides,
                    dilation=1)
                new_space.append(new_dim)
        elif self.op == 'deconv':
            new_space = []
            for i in range(len(space)):
                new_dim = deconv_length(space[i], self.strides, self.kernel_size, self.padding,
                                        output_padding=None)
                new_space.append(new_dim)
        else:
            raise ValueError("Wrong type of operation for capsule")

        return (input_shape[0],) + (self.num_capsule, np.product(self.app_dim) + np.product(self.pos_dim)) + \
               tuple(new_space)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'pos_dim': self.pos_dim,
            'app_dim': self.app_dim,
            'op': self.op,
            'routing_type': self.routing_type,
            'routings': self.routings,
            'strides': self.strides,
            'padding': self.padding,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'coord_add': self.coord_add,
            'is_training': self.is_training,
            'debug_print': self.debug_print
        }
        base_config = super(Caps2dMatwo, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                   num_routing):
    if num_dims == 6:
        votes_t_shape = [5, 0, 1, 2, 3, 4]
        r_t_shape = [1, 2, 3, 4, 5, 0]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(votes, votes_t_shape)
    _, _, _, height, width, caps = votes_trans.get_shape()

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        route = actv.softmax(logits, axis=-1)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = _squash(preactivate)
        activations = activations.write(i, activation)
        act_3d = K.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=-1)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)

    return K.cast(activations.read(num_routing - 1), dtype='float32')


def _squash(input_tensor):
    norm = tf.norm(input_tensor, axis=-1, keepdims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


def matwo_squash(p, axis=-1):
    p_norm_sq = K.sum(K.square(p), axis=axis, keepdims=True)
    p_norm = K.sqrt(p_norm_sq + 1e-9)
    v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
    return v


def Psquash(p, axis=-1):
    v = p / K.max(K.abs(p), axis=axis, keepdims=True)
    return v


def matmult2d(a, b):
    mat = []
    for i in range(a.get_shape()[-2]):
        mat.append(tf.multiply(K.expand_dims(tf.gather(a, i, axis=-2), axis=-1), b))
    c = K.sum(K.stack(mat, axis=-3), axis=-2)
    return c


def routing2d(routing, t_0, u_hat_t_list):
    N, z_1, H_1, W_1, o, t_1 = u_hat_t_list.get_shape().as_list()

    c_t_list = []
    b = K.zeros([N, H_1, W_1, t_0, t_1])
    b_t_list = [K.squeeze(b_t, axis=-1) for b_t in tf.split(b, t_1, axis=-1)]

    u_hat_t_list_ = [K.squeeze(u_hat_t, axis=-1) for u_hat_t in tf.split(u_hat_t_list, t_1, axis=-1)]
    for d in range(routing):

        r_t_mul_u_hat_t_list = []
        for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
            r_t = actv.softmax(b_t, axis=-1)

            if d < routing - 1:
                r_t = K.expand_dims(r_t, axis=1)  # [N, 1, H_1, W_1, t_0]
                r_t_mul_u_hat_t_list.append(
                    K.sum(r_t * u_hat_t, axis=-1))  # sum along the capsule to form the output

            else:
                c_t_list.append(r_t)

        if d < routing - 1:
            p = r_t_mul_u_hat_t_list
            v = matwo_squash(p, axis=2)

            b_t_list_ = []
            idx = 0

            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                v_t1 = K.reshape(tf.gather(v, [idx], axis=0), [N, z_1, H_1, W_1, 1])

                # Evaluate agreement
                rout = K.sum(v_t1 * u_hat_t)
                b_t_list_.append(b_t + rout)
                idx += 1

            b_t_list = b_t_list_
    return c_t_list


def dual_routing2d(routing, t_0, u_hat_t_list, z_pos, z_app):
    N, z_1, H_1, W_1, o, t_1 = u_hat_t_list.get_shape().as_list()

    c_t_list = []
    b = K.zeros([N, H_1, W_1, t_0, t_1])
    b_t_list = [K.squeeze(b_t, axis=-1) for b_t in tf.split(b, t_1, axis=-1)]

    u_hat_t_list_ = [K.squeeze(u_hat_t, axis=-1) for u_hat_t in tf.split(u_hat_t_list, t_1, axis=-1)]
    for d in range(routing):
        r_t_mul_u_hat_t_list = []

        for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
            r_t = actv.sigmoid(b_t)

            if d < routing - 1:
                r_t = K.expand_dims(r_t, axis=1)  # [N, 1, H_1, W_1, t_0]
                r_t_mul_u_hat_t_list.append(
                    K.sum(r_t * u_hat_t, axis=-1))  # sum along the capsule to form the output

            else:
                c_t_list.append(r_t)

        if d < routing - 1:
            p = r_t_mul_u_hat_t_list
            p_pos, p_app = tf.split(p, [z_pos, z_app], axis=2)
            v_app = matwo_squash(p_app, axis=2)
            v_pos = Psquash(p_pos, axis=2)

            b_t_list_ = []
            idx = 0
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                u_hat_pos, u_hat_app = tf.split(u_hat_t, [z_pos, z_app], axis=1)
                v_t1_pos = K.reshape(tf.gather(v_pos, [idx], axis=0), [N, z_pos, H_1, W_1, 1])
                v_t1_app = K.reshape(tf.gather(v_app, [idx], axis=0), [N, z_app, H_1, W_1, 1])

                # Evaluate agreement
                rout = K.sum(u_hat_pos * v_t1_pos, axis=1) * K.sum(u_hat_app * v_t1_app, axis=1)
                b_t_list_.append(b_t + rout)
                idx += 1

            b_t_list = b_t_list_

    return c_t_list


def print_primary_matwocaps_parameters(inputs,
                                       outputs,
                                       capsule_types,
                                       app_dim,
                                       pos_dim,
                                       kernel_size,
                                       name,
                                       is_training,
                                       padding,
                                       strides):
    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = outputs.get_shape().as_list()
    print_string = '{}: ' \
                   'in={} ' \
                   'out={} ' \
                   'caps={} ' \
                   'app_dim={} ' \
                   'pos_dim={} ' \
                   'ks={} ' \
                   's={} ' \
                   'pad={} ' \
                   'train={} ' \
        .format(name,
                inputs_shape,
                outputs_shape,
                capsule_types,
                app_dim,
                pos_dim,
                kernel_size,
                strides,
                padding,
                is_training)
    print(print_string)


def print_matwocaps_parameters(inputs,
                               outputs,
                               routing,
                               routing_type,
                               capsule_types,
                               app_dim,
                               pos_dim,
                               coord_add,
                               kernel_size,
                               name,
                               is_training,
                               padding,
                               strides):
    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = outputs.get_shape().as_list()
    print_string = '{}: ' \
                   'in={} ' \
                   'out={} ' \
                   'rout={} ' \
                   'rout_type={} ' \
                   'caps={} ' \
                   'app_dim={} ' \
                   'pos_dim={} ' \
                   'coord_add={} ' \
                   'ks={} ' \
                   's={} ' \
                   'pad={} ' \
                   'train={} ' \
        .format(name,
                inputs_shape,
                outputs_shape,
                routing,
                routing_type,
                capsule_types,
                app_dim,
                pos_dim,
                coord_add,
                kernel_size,
                strides,
                padding,
                is_training)
    print(print_string)


def mesh2d(shape):
    x, y = np.meshgrid(range(shape[2]), range(shape[1]))
    return y.flatten(), x.flatten()


def coordinate_addition(b, shape):
    y, x = mesh2d(shape)

    coord_add = np.zeros((shape[1] * shape[2], 1, shape[-2], shape[-1]), dtype=np.float32)
    coord_add[:, 0, -1, 0] += x
    coord_add[:, 0, -1, 1] += y
    coord_add[:, 0, -1, 0] /= shape[2]
    coord_add[:, 0, -1, 1] /= shape[1]

    b += (coord_add)
    b = K.reshape(b, shape)
    return b


def l2_normalize_dim(b, axis):
    denom = K.expand_dims(K.sqrt(K.sum(K.square(b), axis=axis)), axis=axis)
    b /= denom
    return b
