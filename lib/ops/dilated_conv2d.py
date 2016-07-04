# Totally untested and never used!

import lib

import theano
import theano.tensor as T
import lasagne

def DilatedConv2D(
    name, 
    input_shape,
    output_dim, 
    filter_size, 
    inputs, 
    he_init=True, 
    dilation=(1,1)
    ):

    input_dim = input_shape[1]

    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    fan_in = input_dim * filter_size**2
    fan_out = output_dim * filter_size**2

    if he_init:
        filters_stdev = np.sqrt(2./fan_in)
    else: # Normalized init (Glorot & Bengio)
        filters_stdev = np.sqrt(2./(fan_in+fan_out))

    W = lib.param(
        name+'.W',
        uniform(
            filters_stdev,
            (input_dim, output_dim, filter_size, filter_size)
        )
    )

    b = lib.param(
        name+'.b',
        np.zeros(output_dim, dtype=theano.config.floatX)
    )

    # Manually apply 'same' padding beforehand
    pad = (filter_size-1)/2

    input_shape = (
        input_shape[0],
        input_shape[1],
        input_shape[2] + pad,
        input_shape[3] + pad
    )

    inputs = lasagne.theano_extensions.padding.pad(
        inputs,
        width=pad,
        batch_ndim=2
    )

    layer = lasagne.layers.DilatedConv2DLayer(
        input_shape, 
        output_dim, 
        filter_size,
        dilation=dilation, 
        pad=0, 
        untie_biases=False,
        W=W, 
        b=b,
        nonlinearity=None, 
        flip_filters=False,
    )

    return layer(inputs)
