import lib
import lib.ops.conv2d
import lib.ops.conv1d
import lib.ops.linear
import lib.ops.batchnorm

import theano.tensor as T
import lasagne

def ConvEncoder(
    name,
    input_n_channels,
    input_size,
    n_pools,
    base_n_filters,
    filter_size,
    output_dim,
    inputs,
    mode='2d', # 1d or 2d,
    deep=False,
    batchnorm=False # seems not to help. debug later.
):

    if mode=='2d':
        conv_fn = lib.ops.conv2d.Conv2D
        bn_axes = [0,2,3]
    elif mode=='1d':
        conv_fn = lib.ops.conv1d.Conv1D
        bn_axes = [0,2]
    else:
        raise Exception()

    # Pad input to the nearest power of two
    new_input_size = 1
    while new_input_size < input_size:
        new_input_size *= 2
    if new_input_size > input_size:
        padding = (new_input_size - input_size) / 2
        inputs = lasagne.theano_extensions.padding.pad(
            inputs,
            width=padding,
            batch_ndim=2
        )
        input_size = new_input_size

    n_filters = base_n_filters

    output = conv_fn(
        name+'.InputConv',
        input_dim=input_n_channels,
        output_dim=n_filters,
        filter_size=filter_size,
        inputs=inputs,
    )

    if batchnorm:
        output = lib.ops.batchnorm.Batchnorm(
            name+'.InputConvBN',
            input_dim=n_filters,
            inputs=output,
            axes=bn_axes
        )

    output = T.nnet.relu(output)

    for i in xrange(n_pools):

        output = conv_fn(
            name+'.Conv{}Strided'.format(i),
            input_dim=n_filters,
            output_dim=2*n_filters,
            filter_size=filter_size,
            inputs=output,
            stride=2
        )

        if batchnorm:
            output = lib.ops.batchnorm.Batchnorm(
                name+'.Conv{}StridedBN'.format(i),
                input_dim=2*n_filters,
                inputs=output,
                axes=bn_axes
            )

        output = T.nnet.relu(output)

        output = conv_fn(
            name+'.Conv{}AfterPool'.format(i),
            input_dim=2*n_filters,
            output_dim=2*n_filters,
            filter_size=filter_size,
            inputs=output,
        )

        if batchnorm:
            output = lib.ops.batchnorm.Batchnorm(
                name+'.Conv{}AfterPoolBN'.format(i),
                input_dim=2*n_filters,
                inputs=output,
                axes=bn_axes
            )

        output = T.nnet.relu(output)

        if deep:
            output = conv_fn(
                name+'.Conv{}AfterPool2'.format(i),
                input_dim=2*n_filters,
                output_dim=2*n_filters,
                filter_size=filter_size,
                inputs=output,
            )

            if batchnorm:
                output = lib.ops.batchnorm.Batchnorm(
                    name+'.Conv{}AfterPool2BN'.format(i),
                    input_dim=2*n_filters,
                    inputs=output,
                    axes=bn_axes
                )

            output = T.nnet.relu(output)

        n_filters *= 2

    if mode=='2d':
        volume = n_filters * (input_size / (2**n_pools))**2
    elif mode=='1d':
        volume = n_filters * (input_size / (2**n_pools))
    else:
        raise Exception()

    output = output.reshape((output.shape[0], volume))

    return lib.ops.linear.Linear(
        name+'.Output',
        volume,
        output_dim,
        output,
        initialization='glorot'
    )