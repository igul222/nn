import lib
import lib.debug
import lib.ops.batchnorm

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import (GpuDnnConvDesc, 
                                        GpuDnnConv, 
                                        GpuDnnConvGradI, 
                                        dnn_conv, 
                                        dnn_pool)

def _deconv2d(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """ 
    from Alec (https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py)
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)

    out = gpu_alloc_empty(
        img.shape[0], 
        kerns.shape[1], 
        img.shape[2]*subsample[0], 
        img.shape[3]*subsample[1]
    )

    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)

    desc = desc(
        out.shape,
        kerns.shape
    )

    d_img = GpuDnnConvGradI()(kerns, img, out, desc)

    return d_img


def Deconv1D(
    name, 
    input_dim, 
    output_dim, 
    filter_size, 
    inputs, 
    stride=2,
    he_init=True
    ):
    """
    inputs: tensor of shape (batch size, num channels, width)
    returns: tensor of shape (batch size, num channels, 2*width)
    """
    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    fan_in = input_dim * filter_size / stride
    fan_out = output_dim * filter_size

    if he_init:
        filters_stdev = np.sqrt(4./(fan_in+fan_out))
    else: # Normalized init (Glorot & Bengio)
        filters_stdev = np.sqrt(2./(fan_in+fan_out))

    # filters_stdev = np.sqrt(1./(input_dim * filter_size))
    # filters_stdev *= np.sqrt(stride) # Because of the stride
    # if he_init:
    #     filters_stdev *= np.sqrt(2.)

    filters = lib.param(
        name+'.Filters',
        uniform(
            filters_stdev,
            (input_dim, output_dim, filter_size, 1)
        )
    )

    biases = lib.param(
        name+'.Biases',
        np.zeros(output_dim, dtype=theano.config.floatX)
    )

    pad = (filter_size-1)/2

    inputs = inputs.dimshuffle(0, 1, 2, 'x')
    result = _deconv2d(
        inputs, 
        filters, 
        subsample=(stride,1),
        border_mode=(pad,0),
    )
    result = T.addbroadcast(result, 3)
    result = result.dimshuffle(0, 1, 2)
    result = result + biases[None, :, None]

    # result = lib.debug.print_stats(name, result)
    # result = lib.ops.batchnorm.Batchnorm(
    #     name+'.BN',
    #     input_dim=output_dim,
    #     inputs=result,
    #     axes=[0,2]
    # )

    return result