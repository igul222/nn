import lib
import lib.ops.conv2d
import lib.ops.conv1d
import lib.ops.linear
import lib.ops.deconv2d
import lib.ops.deconv1d
import lib.ops.batchnorm

import theano.tensor as T

def ConvDecoder(
    name,
    input_dim,
    n_unpools,
    base_n_filters,
    filter_size,
    output_size,
    output_n_channels,
    inputs,
    mode='2d', # 1d or 2d
    deep=False,
    batchnorm=False # seems not to help. debug later.
):

    if mode=='2d':
        conv_fn = lib.ops.conv2d.Conv2D
        deconv_fn = lib.ops.deconv2d.Deconv2D
        bn_axes = [0,2,3]
    elif mode=='1d':
        conv_fn = lib.ops.conv1d.Conv1D
        deconv_fn = lib.ops.deconv1d.Deconv1D
        bn_axes = [0,2]
    else:
        raise Exception()

    # Pad output size to the nearest power of two
    new_output_size = 1
    while new_output_size < output_size:
        new_output_size *= 2
    if new_output_size > output_size:
        padding = (new_output_size - output_size) / 2
        output_size = new_output_size
    else:
        padding = None

    n_filters = base_n_filters * (2**n_unpools)

    if mode=='2d':
        volume = n_filters * (output_size/(2**n_unpools))**2
    elif mode=='1d':
        volume = n_filters * (output_size/(2**n_unpools))
    else:
        raise Exception()

    output = T.nnet.relu(lib.ops.linear.Linear(
        name+'.Input',
        input_dim=input_dim,
        output_dim=volume,
        inputs=inputs,
        initialization='glorot_he'
    ))

    if mode=='2d':
        output = output.reshape((
            output.shape[0],
            n_filters,
            output_size/(2**n_unpools), 
            output_size/(2**n_unpools)
        ))
    elif mode=='1d':
        output = output.reshape((
            output.shape[0],
            n_filters,
            output_size/(2**n_unpools)
        ))
    else:
        raise Exception()

    for i in xrange(n_unpools):
        output = conv_fn(
            name+'.Conv{}BeforeUnpool'.format(i),
            input_dim=n_filters,
            output_dim=n_filters,
            filter_size=filter_size,
            inputs=output,
        )

        if batchnorm:
            output = lib.ops.batchnorm.Batchnorm(
                name+'.Conv{}BeforeUnpoolBN'.format(i),
                input_dim=n_filters,
                inputs=output,
                axes=bn_axes
            )

        output = T.nnet.relu(output)


        if deep:
            output = conv_fn(
                name+'.Conv{}BeforeUnpool2'.format(i),
                input_dim=n_filters,
                output_dim=n_filters,
                filter_size=filter_size,
                inputs=output,
            )

            if batchnorm:
                output = lib.ops.batchnorm.Batchnorm(
                    name+'.Conv{}BeforeUnpool2BN'.format(i),
                    input_dim=n_filters,
                    inputs=output,
                    axes=bn_axes
                )

            output = T.nnet.relu(output)

        output = deconv_fn(
            name+'.Deconv{}'.format(i),
            input_dim=n_filters,
            output_dim=n_filters/2,
            filter_size=filter_size,
            inputs=output
        )

        if batchnorm:
            output = lib.ops.batchnorm.Batchnorm(
                name+'.Deconv{}BN'.format(i),
                input_dim=n_filters/2,
                inputs=output,
                axes=bn_axes
            )

        output = T.nnet.relu(output)

        n_filters /= 2

    output = conv_fn(
        name+'.OutputConv',
        input_dim=n_filters,
        output_dim=output_n_channels,
        filter_size=filter_size,
        inputs=output,
        he_init=False
    )

    if padding is not None:
        if mode == '2d':
            return output[:,:,padding:-padding, padding:-padding]
        elif mode=='1d':
            return output[:,:,padding:-padding]
        else:
            raise Exception()
    else:
        return output