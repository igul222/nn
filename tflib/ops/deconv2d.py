import tflib as lib

import numpy as np
import tensorflow as tf

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def Deconv2D(
    name, 
    input_dim, 
    output_dim, 
    filter_size, 
    inputs, 
    he_init=True,
    weightnorm=None,
    biases=True
    ):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    filters_stdev = np.sqrt(1./(input_dim * filter_size**2))
    filters_stdev *= 2. # Because of the stride
    if he_init:
        filters_stdev *= np.sqrt(2.)

    filter_values = uniform(
        filters_stdev,
        (filter_size, filter_size, output_dim, input_dim)
    )

    filters = lib.param(
        name+'.Filters',
        filter_values
    )

    if weightnorm==None:
        weightnorm = _default_weightnorm
    if weightnorm:
        norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,3)))
        target_norms = lib.param(
            name + '.g',
            norm_values
        )
        norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,3]))
        filters = filters * tf.expand_dims(target_norms / norms, 1)

    input_shape = tf.shape(inputs)
    output_shape = tf.pack([input_shape[0], 2*input_shape[1], 2*input_shape[2], output_dim])

    result = tf.nn.conv2d_transpose(
        value=inputs, 
        filter=filters,
        output_shape=output_shape, 
        strides=[1, 2, 2, 1],
        padding='SAME'
    )

    if biases:
        _biases = lib.param(
            name+'.Biases',
            np.zeros(output_dim, dtype='float32')
        )
        result = tf.nn.bias_add(result, _biases)

    return result