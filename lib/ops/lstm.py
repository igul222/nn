import lib
import lib.ops.linear

import numpy as np
import theano
import theano.tensor as T

def Recurrent(
    name, 
    hidden_dims, 
    step_fn, 
    inputs, 
    non_sequences=[], 
    h0s=None,
    reset=None
    ):

    if not isinstance(inputs, list):
        inputs = [inputs]

    if not isinstance(hidden_dims, list):
        hidden_dims = [hidden_dims]

    if h0s is None:
        h0s = [None]*len(hidden_dims)

    for i in xrange(len(hidden_dims)):
        if h0s[i] is None:
            h0_unbatched = lib.param(
                name + '.h0_' + str(i),
                np.zeros((hidden_dims[i],), dtype=theano.config.floatX)
            )
            num_batches = inputs[0].shape[1]
            h0s[i] = T.alloc(h0_unbatched, num_batches, hidden_dims[i])

        h0s[i] = T.patternbroadcast(h0s[i], [False] * h0s[i].ndim)

    if reset is not None:
        last_hiddens = []
        for i in xrange(len(h0s)):
            # The shape of last_hidden doesn't matter right now; we assume
            # it won't be used until we put something proper in it.
            last_hidden = theano.shared(
                np.zeros([1]*h0s[i].ndim, dtype=h0s[i].dtype),
                name=name+'.last_hidden_'+str(i)
            )
            last_hiddens.append(last_hidden)
            h0s[i] = theano.ifelse.ifelse(reset, h0s[i], last_hidden)

    outputs, _ = theano.scan(
        step_fn,
        sequences=inputs,
        outputs_info=h0s,
        non_sequences=non_sequences
    )

    if reset is not None:
        if len(last_hiddens) == 1:
            last_hiddens[0].default_update = outputs[-1]
        else:
            for i in xrange(len(last_hiddens)):
                last_hiddens[i].default_update = outputs[i][-1]

    return outputs

def GRUStep(name, input_dim, hidden_dim, current_input, last_hidden):
    processed_input = lib.ops.linear.Linear(
        name+'.Input',
        input_dim,
        3 * hidden_dim,
        current_input
    )

    gates = T.nnet.sigmoid(
        lib.ops.linear.Linear(
            name+'.Recurrent_Gates',
            hidden_dim,
            2 * hidden_dim,
            last_hidden,
            biases=False
        ) + processed_input[:, :2*hidden_dim]
    )

    update = gates[:, :hidden_dim]
    reset  = gates[:, hidden_dim:]

    scaled_hidden = reset * last_hidden

    candidate = T.tanh(
        lib.ops.linear.Linear(
            name+'.Recurrent_Candidate', 
            hidden_dim, 
            hidden_dim, 
            scaled_hidden,
            biases=False,
            initialization='orthogonal'
        ) + processed_input[:, 2*hidden_dim:]
    )

    one = lib.floatX(1.0)
    return (update * candidate) + ((one - update) * last_hidden)

def LowMemGRU(name, input_dim, hidden_dim, inputs, h0=None, reset=None):
    inputs = inputs.dimshuffle(1,0,2)

    def step(current_input, last_hidden):
        return GRUStep(
            name+'.Step', 
            input_dim, 
            hidden_dim, 
            current_input, 
            last_hidden
        )

    if h0 is None:
        h0s = None
    else:
        h0s = [h0]

    out = Recurrent(
        name+'.Recurrent',
        hidden_dim,
        step,
        inputs,
        h0s=h0s,
        reset=reset
    )

    out = out.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out

def GRU(name, input_dim, hidden_dim, inputs, h0=None, reset=None):
    inputs = inputs.dimshuffle(1,0,2)

    processed_inputs = lib.ops.linear.Linear(
        name+'.Input',
        input_dim,
        3 * hidden_dim,
        inputs,
        biases=True
        # biases=(not batchnorm)
    )

    # if batchnorm:
    #     processed_inputs = lib.ops.BatchNormalize(
    #         name+'.BatchNormalize',
    #         3 * hidden_dim,
    #         processed_inputs,
    #         stepwise=True
    #     )

    def step(current_processed_input, last_hidden):
        gates = T.nnet.sigmoid(
            lib.ops.linear.Linear(
                name+'.Recurrent_Gates', 
                hidden_dim, 
                2 * hidden_dim, 
                last_hidden,
                biases=False
            ) + current_processed_input[:, :2*hidden_dim]
        )

        update = gates[:, :hidden_dim]
        reset  = gates[:, hidden_dim:]

        scaled_hidden = reset * last_hidden

        candidate = T.tanh(
            lib.ops.linear.Linear(
                name+'.Recurrent_Candidate', 
                hidden_dim, 
                hidden_dim, 
                scaled_hidden,
                biases=False,
                initialization='orthogonal'
            ) + current_processed_input[:, 2*hidden_dim:]
        )

        one = lib.floatX(1.0)
        return (update * candidate) + ((one - update) * last_hidden)

    if h0 is None:
        h0s = None
    else:
        h0s = [h0]

    out = Recurrent(
        name+'.Recurrent',
        hidden_dim,
        step,
        processed_inputs,
        reset=reset,
        h0s=h0s
    )
    out = out.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out


def LSTM(name, input_dim, hidden_dim, inputs, h0s=None, reset=None):
    inputs = inputs.dimshuffle(1,0,2)

    processed_inputs = lib.ops.linear.Linear(
        name+'.Input',
        input_dim,
        4 * hidden_dim,
        inputs,
        biases=True,
        # weightnorm=False
        # biases=(not batchnorm)
    )

    def step(current_processed_input, last_hidden, last_cell):
        gates = lib.ops.linear.Linear(
            name+'.Recurrent_Gates', 
            hidden_dim, 
            4 * hidden_dim, 
            last_hidden,
            biases=False,
            initialization='orthogonal',
            # weightnorm=False
        ) + current_processed_input

        # # Layer normalization
        # means = gates.mean(axis=1, keepdims=True)
        # variances = gates.var(axis=1, keepdims=True)
        # scale = lib.param(name+'.scale', np.ones(4*hidden_dim, dtype='float32')).dimshuffle('x',0)
        # offset = lib.param(name+'.offset', np.zeros(4*hidden_dim, dtype='float32')).dimshuffle('x',0)
        # stdevs = T.sqrt(variances + lib.floatX(1e-5))
        # gates = T.nnet.bn.batch_normalization(
        #     gates, 
        #     scale, 
        #     offset, 
        #     means, 
        #     stdevs, 
        # )

        i, z, o, f = gates[:,::4], gates[:,1::4], gates[:,2::4], gates[:,3::4]

        i = T.nnet.sigmoid(i)
        z = T.tanh(z)
        o = T.nnet.sigmoid(o)
        f = T.nnet.sigmoid(f + 1)

        z = (i*z) + (f*last_cell)
        h = T.tanh(z)*o

        return (h,z)

    out, out_c = Recurrent(
        name+'.Recurrent',
        [hidden_dim, hidden_dim],
        step,
        processed_inputs,
        reset=reset,
        h0s=h0s
    )

    out = out.dimshuffle(1,0,2)
    out_c = out_c.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out, out_c
