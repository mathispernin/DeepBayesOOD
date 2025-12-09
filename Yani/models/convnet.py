from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import (
    Dense,
    Activation,
    Flatten,
    Lambda,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    ELU,
    PReLU
)
import numpy as np

np.random.seed(0)
tf.set_random_seed(0)

alpha = 0.2
p = 0.3

def Dropout(p):
    # Keras Dropout via Lambda for TF1-style compatibility
    layer = Lambda(lambda x: K.dropout(x, p), output_shape=lambda shape: shape)
    return layer

def construct_filter_shapes(layer_channels, filter_width=5):
    filter_shapes = []
    for n_channel in layer_channels:
        shape = (n_channel, filter_width, filter_width)
        filter_shapes.append(shape)
    return filter_shapes

def ConvNet(name, input_shape, filter_shapes, fc_layer_sizes,
            activation='relu', batch_norm=False, last_activation=None,
            weight_init='glorot_normal', subsample=None, dropout=False):
    """
    Construct a deep convolutional network in TF1-compatible Keras
    """

    num_conv_layers = len(filter_shapes)
    num_fc_layers = len(fc_layer_sizes)

    if last_activation is None:
        last_activation = activation
    if subsample is None:
        subsample = [(2, 2) for _ in range(num_conv_layers)]

    model = Sequential()
    conv_output_shape = []
    bias_flag = (not batch_norm)

    with tf.variable_scope(name):
        # Convolutional layers
        for l in range(num_conv_layers):
            n_channel, height, width = filter_shapes[l]

            strides = subsample[l] if subsample else (1, 1)

            # First layer can have input shape
            if l == 0:
                model.add(Conv2D(
                    filters=n_channel,
                    kernel_size=(height, width),
                    strides=strides,
                    padding='same',
                    kernel_initializer=weight_init,
                    use_bias=bias_flag,
                    input_shape=input_shape
                ))
            else:
                model.add(Conv2D(
                    filters=n_channel,
                    kernel_size=(height, width),
                    strides=strides,
                    padding='same',
                    kernel_initializer=weight_init,
                    use_bias=bias_flag
                ))

            conv_output_shape.append(model.output_shape[1:])

            if batch_norm:
                model.add(BatchNormalization())  # mode param removed in TF2 Keras

            if dropout:
                model.add(Dropout(p))

            # Activation
            if activation == 'lrelu':
                model.add(LeakyReLU(alpha=alpha))
            elif activation == 'elu':
                model.add(ELU(alpha=1.0))
            elif activation == 'prelu':
                model.add(PReLU())
            else:
                model.add(Activation(activation))

        # Flatten
        flatten_fn = lambda x: tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        model.add(Lambda(flatten_fn, name='flatten'))

        # Fully connected layers
        for l in range(num_fc_layers):
            is_last = (l + 1 == num_fc_layers)
            layer_bias = True if is_last else bias_flag

            model.add(Dense(
                units=fc_layer_sizes[l],
                kernel_initializer=weight_init,
                use_bias=layer_bias,
                name=f'dense{l}'
            ))

            if batch_norm and not is_last:
                model.add(BatchNormalization())

            if dropout and not is_last:
                model.add(Dropout(p))

            # Activation
            layer_activation = last_activation if is_last else activation
            if layer_activation == 'lrelu':
                model.add(LeakyReLU(alpha=alpha))
            elif layer_activation == 'elu':
                model.add(ELU(alpha=1.0))
            elif layer_activation == 'prelu':
                model.add(PReLU())
            else:
                model.add(Activation(layer_activation))

    return model, conv_output_shape


