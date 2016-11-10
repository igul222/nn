import tflib as lib
import tflib.ops.linear

import numpy as np
import tensorflow as tf

class GRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, inputs, state, scope=None):
        gates = tf.nn.sigmoid(
            lib.ops.linear.Linear(
                self._name+'.Gates',
                self._n_in + self._n_hid,
                2 * self._n_hid,
                tf.concat(1, [inputs, state])
            )
        )

        update, reset = tf.split(1, 2, gates)
        scaled_state = reset * state

        candidate = tf.tanh(
            lib.ops.linear.Linear(
                self._name+'.Candidate', 
                self._n_in + self._n_hid, 
                self._n_hid, 
                tf.concat(1, [inputs, scaled_state])
            )
        )

        output = (update * candidate) + ((1 - update) * state)

        return output, output

def GRU(name, n_in, n_hid, inputs):
    h0 = lib.param(name+'.h0', np.zeros(n_hid, dtype='float32'))
    batch_size = tf.shape(inputs)[0]
    h0 = tf.reshape(tf.tile(h0, tf.pack([batch_size])), tf.pack([batch_size, n_hid]))
    return tf.nn.dynamic_rnn(GRUCell(name, n_in, n_hid), inputs, initial_state=h0, swap_memory=True)[0]

# class GRUCell(tf.nn.rnn_cell.RNNCell):
#     def __init__(self, name, n_in, n_hid):
#         self._n_in = n_in
#         self._n_hid = n_hid
#         self._name = name

#     @property
#     def state_size(self):
#         return self._n_hid

#     @property
#     def output_size(self):
#         return self._n_hid

#     def __call__(self, processed_inputs, state, scope=None):
#         # pi_update, pi_reset, pi_candidate = tf.split(1, 3, processed_inputs)

#         gates = tf.nn.sigmoid(
#             lib.ops.linear.Linear(
#                 self._name+'.Gates_R',
#                 self._n_hid,
#                 2 * self._n_hid,
#                 state,
#                 biases=False
#             ) + processed_inputs[:,:2*self._n_hid]
#         )

#         update, reset = tf.split(1, 2, gates)
#         scaled_state = reset * state

#         candidate = tf.tanh(
#             lib.ops.linear.Linear(
#                 self._name+'.Candidate_R', 
#                 self._n_hid, 
#                 self._n_hid, 
#                 scaled_state
#                 # tf.concat(1, [inputs, scaled_state])
#             ) + processed_inputs[:,2*self._n_hid:]
#         )

#         output = (update * candidate) + ((1 - update) * state)

#         return output, output

# def GRU(name, n_in, n_hid, inputs):
#     processed_inputs = lib.ops.linear.Linear(name+'.Inputs', n_in, 3*n_hid, inputs)
#     h0 = lib.param(name+'.h0', np.zeros(n_hid, dtype='float32'))
#     batch_size = tf.shape(inputs)[0]
#     h0 = tf.reshape(tf.tile(h0, tf.pack([batch_size])), tf.pack([batch_size, n_hid]))
#     return tf.nn.dynamic_rnn(GRUCell(name, n_in, n_hid), processed_inputs, initial_state=h0, swap_memory=True)[0]