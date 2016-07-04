import lib
import lib.debug

import numpy as np
import theano
import theano.tensor as T
import lasagne

def Conv1D(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, stride=1, border_mode='half'):
    """
    inputs.shape: (batch size, input_dim, height)
    output.shape: (batch size, output_dim, height)
    """
    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    fan_in = input_dim * filter_size
    fan_out = output_dim * filter_size
    fan_out /= stride

    if he_init:
        filters_stdev = np.sqrt(4./(fan_in+fan_out))
    else: # Normalized init (Glorot & Bengio)
        filters_stdev = np.sqrt(2./(fan_in+fan_out))

    filters = lib.param(
        name+'.Filters',
        uniform(
            filters_stdev,
            (output_dim, input_dim, filter_size, 1)
        )
    )

    inputs = inputs.dimshuffle(0, 1, 2, 'x')
    result = T.nnet.conv2d(
        inputs, 
        filters, 
        border_mode=border_mode,
        subsample=(stride, 1)
    )
    result = T.addbroadcast(result, 3)
    result = result.dimshuffle(0, 1, 2)

    if biases:
        biases_ = lib.param(
            name+'.Biases',
            np.zeros(output_dim, dtype=theano.config.floatX)
        )
        result = result + biases_[None, :, None]

    # result = lib.debug.print_shape(name, result)
    return result