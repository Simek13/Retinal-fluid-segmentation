'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the network definitions for the various capsule network architectures.
'''

from tensorflow.keras import layers, models, Model
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')

from capsule_layers import PrimaryCaps2dMatwo, Caps2dMatwo, \
    DualLength


class MatwoCapsNet(Model):

    def __init__(self, num_labels=6, routing_type='dual', routing=3, data_format='channels_last', name='matwo_capsnet',
                 **kwargs):
        super(MatwoCapsNet, self).__init__(name=name, **kwargs)
        self.pos_dim = [4, 4]
        self.app_dim = [5, 5]
        self.level_caps = [5, 5, 6, 7]
        self.padding = 'same'
        self.coord_add = False
        self.num_labels = num_labels
        self.routing_type = routing_type
        self.routing = routing
        self.data_format = data_format
        self.primary_caps = PrimaryCaps2dMatwo(pos_dim=self.pos_dim, app_dim=self.app_dim,
                                               num_capsule=int(self.level_caps[0]), kernel_size=5,
                                               strides=1, name='primary_caps', padding=self.padding,
                                               data_format=data_format)
        self.conv_cap_1_2 = Caps2dMatwo(routings=1, routing_type=routing_type, pos_dim=self.pos_dim,
                                        app_dim=self.app_dim,
                                        num_capsule=int(self.level_caps[1]), kernel_size=5, name='caps_1df_cd1',
                                        coord_add=self.coord_add,
                                        padding=self.padding, strides=1, op="conv")
        self.conv_cap_2_1 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                        app_dim=self.app_dim,
                                        num_capsule=int(self.level_caps[1]), kernel_size=5, name='caps_12_cd1',
                                        coord_add=self.coord_add,
                                        padding=self.padding, strides=2, op="conv")
        self.conv_cap_2_2 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                        app_dim=self.app_dim,
                                        num_capsule=int(self.level_caps[1]), kernel_size=5, name='caps_12_cd2',
                                        coord_add=self.coord_add,
                                        padding=self.padding, strides=1, op="conv")
        self.conv_cap_3_1 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                        app_dim=self.app_dim,
                                        num_capsule=int(self.level_caps[2]), kernel_size=5, name='caps_14_cd1',
                                        coord_add=self.coord_add,
                                        padding=self.padding, strides=2, op="conv")
        self.conv_cap_3_2 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                        app_dim=self.app_dim,
                                        num_capsule=int(self.level_caps[2]), kernel_size=5, name='caps_14_cd2',
                                        coord_add=self.coord_add,
                                        padding=self.padding, strides=1, op="conv")
        self.conv_cap_4_1 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                        app_dim=self.app_dim,
                                        num_capsule=int(self.level_caps[3]), kernel_size=5, name='caps_18_cd1',
                                        coord_add=self.coord_add,
                                        padding=self.padding, strides=2, op="conv")
        self.conv_cap_4_2 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                        app_dim=self.app_dim,
                                        num_capsule=int(self.level_caps[3]), kernel_size=5, name='caps_18_cd2',
                                        coord_add=self.coord_add,
                                        padding=self.padding, strides=1, op="conv")
        self.conv_cap_4_3 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                        app_dim=self.app_dim,
                                        num_capsule=int(self.level_caps[3]), kernel_size=5, name='caps_18_cd3',
                                        coord_add=self.coord_add,
                                        padding=self.padding, strides=1, op="conv")
        self.deconv_cap_1_1 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                          app_dim=self.app_dim,
                                          num_capsule=int(self.level_caps[3]), kernel_size=4, name='caps_14_du1',
                                          coord_add=self.coord_add,
                                          padding=self.padding, strides=2, op="deconv")
        self.deconv_cap_1_2 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                          app_dim=self.app_dim,
                                          num_capsule=int(self.level_caps[3]), kernel_size=5, name='caps_14_cu2',
                                          coord_add=self.coord_add,
                                          padding=self.padding, strides=1, op="conv")
        self.deconv_cap_2_1 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                          app_dim=self.app_dim,
                                          num_capsule=int(self.level_caps[2]), kernel_size=4, name='caps_12_du1',
                                          coord_add=self.coord_add,
                                          padding=self.padding, strides=2, op="deconv")
        self.deconv_cap_2_2 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                          app_dim=self.app_dim,
                                          num_capsule=int(self.level_caps[2]), kernel_size=5, name='caps_12_cu2',
                                          coord_add=self.coord_add,
                                          padding=self.padding, strides=1, op="conv")
        self.deconv_cap_3_1 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                          app_dim=self.app_dim,
                                          num_capsule=int(self.level_caps[2]), kernel_size=4, name='caps_1_du1',
                                          coord_add=self.coord_add,
                                          padding=self.padding, strides=2, op="deconv")
        self.deconv_cap_3_2 = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=self.pos_dim,
                                          app_dim=self.app_dim,
                                          num_capsule=num_labels, kernel_size=1, name='caps_1_c2',
                                          coord_add=self.coord_add, padding=self.padding,
                                          strides=1, op="conv")
        self.up_1 = layers.Concatenate(axis=1, name='up_1')
        self.up_2 = layers.Concatenate(axis=1, name='up_2')
        self.up_3 = layers.Concatenate(axis=1, name='up_3')
        self.dual_length = DualLength(pos_dim=self.pos_dim, app_dim=self.app_dim)

    def call(self, inputs):
        x = self.primary_caps(inputs)
        x = self.conv_cap_1_2(x)
        skip1 = x

        # 1/2
        x = self.conv_cap_2_1(x)
        x = self.conv_cap_2_2(x)
        skip2 = x

        # 1/4
        x = self.conv_cap_3_1(x)
        x = self.conv_cap_3_2(x)
        skip3 = x

        # 1/8
        x = self.conv_cap_4_1(x)
        x = self.conv_cap_4_2(x)
        x = self.conv_cap_4_3(x)

        # 1/4
        x = self.deconv_cap_1_1(x)
        x = self.up_1([x, skip3])
        x = self.deconv_cap_1_2(x)

        # 1/2
        x = self.deconv_cap_2_1(x)
        x = self.up_2([x, skip2])
        x = self.deconv_cap_2_2(x)

        # 1
        x = self.deconv_cap_3_1(x)
        x = self.up_3([x, skip1])
        x = self.deconv_cap_3_2(x)

        prediction = self.dual_length(x)
        return prediction


def Matwo_CapsNet(input_shape, num_labels=6, routing_type='dual', routing=3,
                  data_format='channels_last'):
    padding = 'same'
    coord_add = True
    pos_dim = [4, 4]
    app_dim = [5, 5]
    level_caps = [5, 5, 6, 7]

    input_tensor = layers.Input(batch_shape=input_shape)

    x = PrimaryCaps2dMatwo(pos_dim=pos_dim, app_dim=app_dim, num_capsule=int(level_caps[0]), kernel_size=5,
                           strides=1, name='primary_caps', padding=padding, data_format=data_format)(input_tensor)

    x = Caps2dMatwo(routings=1, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[1]), kernel_size=5, name='caps_1df_cd1', coord_add=coord_add,
                    padding=padding, strides=1, op="conv")(x)
    skip1 = x
    # 1/2
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[1]), kernel_size=5, name='caps_12_cd1', coord_add=coord_add,
                    padding=padding, strides=2, op="conv")(x)

    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[1]), kernel_size=5, name='caps_12_cd2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv")(x)
    skip2 = x

    # 1/4
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=5, name='caps_14_cd1', coord_add=coord_add,
                    padding=padding, strides=2, op="conv")(x)
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=5, name='caps_14_cd2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv")(x)
    skip3 = x

    # 1/8
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[3]), kernel_size=5, name='caps_18_cd1', coord_add=coord_add,
                    padding=padding, strides=2, op="conv")(x)
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[3]), kernel_size=5, name='caps_18_cd2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv")(x)
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[3]), kernel_size=5, name='caps_18_cd3', coord_add=coord_add,
                    padding=padding, strides=1, op="conv")(x)

    # 1/4
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=4, name='caps_14_du1', coord_add=coord_add,
                    padding=padding, strides=2, op="deconv")(x)
    x = layers.Concatenate(axis=1, name='up_1')([x, skip3])
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=5, name='caps_14_cu2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv")(x)

    # 1/2
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=4, name='caps_12_du1', coord_add=coord_add,
                    padding=padding, strides=2, op="deconv")(x)
    x = layers.Concatenate(axis=1, name='up_2')([x, skip2])
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=5, name='caps_12_cu2', coord_add=coord_add,
                    padding=padding, strides=1, op="conv")(x)

    # 1
    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=int(level_caps[2]), kernel_size=4, name='caps_1_du1', coord_add=coord_add,
                    padding=padding, strides=2, op="deconv")(x)
    x = layers.Concatenate(axis=1, name='up_3')([x, skip1])

    x = Caps2dMatwo(routings=routing, routing_type=routing_type, pos_dim=pos_dim, app_dim=app_dim,
                    num_capsule=num_labels, kernel_size=1, name='caps_1_c2', coord_add=coord_add, padding=padding,
                    strides=1, op="conv")(x)

    prediction = DualLength(pos_dim=pos_dim, app_dim=app_dim)(x)

    model = models.Model(inputs=input_tensor, outputs=prediction)

    return [model]


