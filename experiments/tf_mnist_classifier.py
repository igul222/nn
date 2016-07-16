import os, sys
sys.path.append(os.getcwd())

# try: # This only matters on Ishaan's computer
#     import experiment_tools
#     experiment_tools.wait_for_gpu(tf=True)
# except ImportError:
#     pass

import tflib as lib
import tflib.train_loop
import tflib.mnist
import tflib.ops.mlp

import numpy as np
import tensorflow as tf

import functools

LR = 1e-3
BATCH_SIZE = 10000

DEVICES = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

TIMES = {
    'mode': 'iters',
    'print_every': 1*5,
    'stop_after': 10*5,
}

lib.print_model_settings(locals().copy())

with tf.Session() as session:

    inputs = tf.placeholder(tf.float32, shape=[None, 784])
    targets = tf.placeholder(tf.int32, shape=[None])

    tower_costs = []
    tower_accs = []

    split_inputs = tf.split(0, len(DEVICES), inputs)
    split_targets = tf.split(0, len(DEVICES), targets)

    for device, input_split, target_split in zip(DEVICES, split_inputs, split_targets):
        with tf.device(device):

            logits = lib.ops.mlp.MLP(
                'MLP',
                input_dim=784,
                hidden_dim=2048,
                output_dim=10,
                n_layers=7,
                inputs=input_split
            )


            cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target_split)
            )

            acc = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.to_int32(tf.argmax(logits, dimension=1)),
                        target_split
                    ),
                    tf.float32
                )
            )

            tower_costs.append(cost)
            tower_accs.append(acc)

        cost = tf.reduce_mean(tf.concat(0, [tf.expand_dims(c, 0) for c in tower_costs]), 0)
        acc = tf.reduce_mean(tf.concat(0, [tf.expand_dims(a, 0) for a in tower_accs]), 0)

    train_data, dev_data, test_data = lib.mnist.load(
        BATCH_SIZE,
        BATCH_SIZE
    )

    lib.train_loop.train_loop(
        session=session,
        inputs=[inputs, targets],
        cost=cost,
        prints=[
            ('acc', acc)
        ],
        optimizer=tf.train.AdamOptimizer(LR),
        train_data=train_data,
        test_data=dev_data,
        times=TIMES
    )