import lib

import theano
import numpy as np

def Embedding(name, n_symbols, output_dim, inputs):
    vectors = lib.param(
        name,
        np.random.randn(
            n_symbols, 
            output_dim
        ).astype(theano.config.floatX)
    )

    output_shape = [
        inputs.shape[i]
        for i in xrange(inputs.ndim)
    ] + [output_dim]

    return vectors[inputs.flatten()].reshape(output_shape)