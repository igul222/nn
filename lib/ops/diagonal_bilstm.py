import lib
import lib.ops.conv1d
import lib.ops.conv2d

import numpy as np
import theano
import theano.tensor as T

def _skew(height, width, inputs):
    """
    input.shape: (batch size, dim, height, width)
    """
    buffer = T.zeros(
        (inputs.shape[0], inputs.shape[1], height, 2*width - 1),
        theano.config.floatX
    )

    for i in xrange(height):
        buffer = T.inc_subtensor(buffer[:, :, i, i:i+width], inputs[:,:,i,:])

    return buffer

def _unskew(height, width, padded):
    """
    padded.shape: (batch size, dim, height, 2*width - 1)
    """
    return T.stack([padded[:, :, i, i:i+width] for i in xrange(height)], axis=2)

def DiagonalLSTM(name, input_dim, output_dim, input_shape, inputs):
    """
    inputs_shape: (n_channels, height, width)
    inputs.shape: (batch size, input_dim, height, width)
    outputs.shape: (batch size, output_dim, height, width)
    """
    n_channels, height, width = input_shape

    inputs = _skew(height, width, inputs)

    # TODO benchmark running skew after input_to_state, might be faster
    input_to_state = lib.ops.conv2d.Conv2D(
        name+'.InputToState', 
        input_dim, 
        4*output_dim, 
        1, inputs, 
        mask_type=('b', n_channels), 
        he_init=False
    )

    batch_size = inputs.shape[0]

    c0_unbatched = lib.param(
        name + '.c0',
        np.zeros((output_dim, height), dtype=theano.config.floatX)
    )
    c0 = T.alloc(c0_unbatched, batch_size, output_dim, height)

    h0_unbatched = lib.param(
        name + '.h0',
        np.zeros((output_dim, height), dtype=theano.config.floatX)
    )
    h0 = T.alloc(h0_unbatched, batch_size, output_dim, height)

    def step_fn(current_input_to_state, prev_c, prev_h):
        # all args have shape (batch size, output_dim, height)

        # TODO consider learning this padding
        prev_h_padded = T.zeros((batch_size, output_dim, 1+height), dtype=theano.config.floatX)
        prev_h_padded = T.inc_subtensor(prev_h_padded[:,:,1:], prev_h)

        state_to_state = lib.ops.conv1d.Conv1D(
            name+'.StateToState', 
            output_dim, 
            4*output_dim, 
            2, 
            prev_h_padded, 
            biases=False,
            he_init=False,
            border_mode='valid'
        )

        gates = current_input_to_state + state_to_state

        o_f_i = T.nnet.sigmoid(gates[:,:3*output_dim,:])
        o = o_f_i[:,0*output_dim:1*output_dim,:]
        f = o_f_i[:,1*output_dim:2*output_dim,:]
        i = o_f_i[:,2*output_dim:3*output_dim,:]
        g = T.tanh(gates[:,3*output_dim:4*output_dim,:])

        new_c = (f * prev_c) + (i * g)
        new_h = o * T.tanh(new_c)

        return (new_c, new_h)

    outputs, _ = theano.scan(
        step_fn,
        sequences=input_to_state.dimshuffle(3,0,1,2),
        outputs_info=[c0, h0]
    )
    all_cs = outputs[0].dimshuffle(1,2,3,0)
    all_hs = outputs[1].dimshuffle(1,2,3,0)

    return _unskew(height, width, all_hs)

def DiagonalBiLSTM(name, input_dim, output_dim, input_shape, inputs):
    """
    inputs.shape: (batch size, input_dim, inputs_shape)
    result.shape: (batch size, output_dim, inputs_shape)
    """
    forward = DiagonalLSTM(name+'.Forward', input_dim, output_dim, input_shape, inputs)
    backward = DiagonalLSTM(name+'.Backward', input_dim, output_dim, input_shape, inputs[:,:,:,::-1])[:,:,:,::-1]
    return T.inc_subtensor(forward[:,:,1:,:], backward[:,:,:-1,:])