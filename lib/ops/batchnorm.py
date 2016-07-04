import lib

import numpy as np
import theano
import theano.tensor as T

def Batchnorm(
    name, 
    input_dim, 
    inputs, 
    stepwise=False, 
    axes=None, 
    wrt=None, 
    cc=False, 
    i_gamma=None,
    i_beta=None
):
    if wrt is None:
        wrt = inputs

    if axes:
        means = wrt.mean(axis=axes, keepdims=True)
        variances = wrt.var(axis=axes, keepdims=True)
    # elif stepwise:
    #     means = wrt.mean(axis=1, keepdims=True)
    #     variances = wrt.var(axis=1, keepdims=True)
    else:
        means = wrt.reshape((-1, input_dim)).mean(axis=0)
        variances = wrt.reshape((-1, input_dim)).var(axis=0)

    # if cc:
    #     means = theano.gradient.zero_grad(means)
    #     variances = theano.gradient.zero_grad(variances)

    if i_gamma is None:
        i_gamma = lib.floatX(0.1) * np.ones(input_dim, dtype=theano.config.floatX)

    if i_beta is None:
        i_beta = np.zeros(input_dim, dtype=theano.config.floatX)

    gamma = lib.param(
        name + '.gamma',
        i_gamma
    )

    beta = lib.param(
        name + '.beta',
        i_beta
    )

    stdevs = T.sqrt(variances + lib.floatX(1e-6))

    stdevs.name = name+'.stdevs'
    means.name = name+'.means'

    # return (((inputs - means) / stdevs) * gamma) + beta
    if axes:
        dimshuffle_pattern = [
            'x' if i in axes else 0 
            for i in xrange(inputs.ndim)
        ]
        return T.nnet.bn.batch_normalization(
            inputs, 
            gamma.dimshuffle(*dimshuffle_pattern), 
            beta.dimshuffle(*dimshuffle_pattern), 
            means, 
            stdevs, 
            mode='low_mem'
        )
    else:
        return T.nnet.bn.batch_normalization(
            inputs, 
            gamma.dimshuffle('x',0), 
            beta.dimshuffle('x',0), 
            means.dimshuffle('x',0), 
            stdevs.dimshuffle('x',0), 
            mode='low_mem'
        )
