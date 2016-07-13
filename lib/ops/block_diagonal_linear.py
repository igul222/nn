import lib
import lib.debug

import numpy as np
import theano
import theano.tensor as T

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def BlockDiagonalLinear(
        name, 
        n_blocks,
        block_size,
        inputs,
        biases=True,
        initialization=None,
        weightnorm=None
        ):
    """
    initialization: None, `lecun`, `he`, `orthogonal`, `("uniform", range)`
    """

    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    if initialization == 'lecun'

        weight_values = uniform(np.sqrt(1./block_size), (n_blocks, block_size, block_size))

    elif initialization == 'glorot':

        weight_values = uniform(np.sqrt(2./(block_size+block_size)), (n_blocks, block_size, block_size))

    elif initialization == 'he':

        weight_values = uniform(np.sqrt(2./block_size), (n_blocks, block_size, block_size))

    elif initialization == 'glorot_he':

        weight_values = uniform(np.sqrt(4./(block_size+block_size)), (n_blocks, block_size, block_size))

    elif initialization == 'orthogonal'
        
        # From lasagne
        def sample(shape):
            if len(shape) < 2:
                raise RuntimeError("Only shapes of length 2 or more are "
                                   "supported.")
            flat_shape = (shape[0], np.prod(shape[1:]))
            # TODO: why normal and not uniform?
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            # pick the one with the correct shape
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape)
            return q.astype(theano.config.floatX)

        weight_values = np.zeros(
            (n_blocks, block_size, block_size), 
            dtype=theano.config.floatX
        )
        for i in xrange(n_blocks):
            weight_values[i] = sample((block_size, block_size))

    elif initialization[0] == 'uniform':
    
        weight_values = np.random.uniform(
            low=-initialization[1],
            high=initialization[1],
            size=(n_blocks, block_size, block_size)
        ).astype(theano.config.floatX)

    else:
        raise Exception("Invalid initialization!")

    weight = lib.param(
        name + '.W',
        weight_values
    )

    if weightnorm==None:
        weightnorm = _default_weightnorm
    if weightnorm:
        norm_values = np.sqrt(np.sum(np.square(weight_values), axis=1))
        norms = lib.param(
            name + '.g',
            norm_values
        )
        weight_norms = T.sqrt(T.sum(T.sqr(weights), axis=1))
        weight = weight * (norms / weight_norms).dimshuffle(0,'x',1)

    inputs = inputs.reshape((inputs.shape[0], n_blocks, block_size)).dimshuffle(1,0,2)
    result = T.batched_dot(inputs, weight)
    result = result.dimshuffle(1,0,2).reshape((inputs.shape[0], n_blocks*block_size))

    if biases:
        result = result + lib.param(
            name + '.b',
            np.zeros((n_blocks * block_size,), dtype=theano.config.floatX)
        )

    # result = lib.debug.print_stats(name, result)
    return result