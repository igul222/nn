import tflib as lib
import tflib.debug

import numpy as np
import tensorflow as tf

def Batchnorm(name, axes, inputs):
    if axes == [0,2,3]:
        inputs = tf.transpose(inputs, [0,2,3,1])

        mean, var = tf.nn.moments(inputs, [0,1,2], keep_dims=False)
        offset = lib.param(name+'.offset', np.zeros(mean.get_shape()[-1], dtype='float32'))
        scale = lib.param(name+'.scale', np.ones(var.get_shape()[-1], dtype='float32'))
        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)

        return tf.transpose(result, [0,3,1,2])
    else:
        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        offset = lib.param(name+'.offset', np.zeros(mean.get_shape(), dtype='float32'))
        scale = lib.param(name+'.scale', np.ones(var.get_shape(), dtype='float32'))
        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)
        # lib.debug.print_stats(name, result)
        return result