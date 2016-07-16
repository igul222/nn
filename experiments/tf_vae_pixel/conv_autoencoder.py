import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True)
except ImportError:
    pass

import tflib as lib
import tflib.train_loop
import tflib.mnist
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear

import numpy as np
import tensorflow as tf

import functools

LR = 1e-3
BATCH_SIZE = 100

DEVICES = ['/gpu:0']

TIMES = {
    'mode': 'iters',
    'print_every': 1*500,
    'stop_after': 10*500,
}

lib.print_model_settings(locals().copy())

with tf.Session() as session:

    inputs = tf.placeholder(tf.float32, shape=[None, 784])
    targets = tf.placeholder(tf.int32, shape=[None])

    output = tf.reshape(inputs, [-1, 28, 28, 1])

    output = tf.nn.relu(lib.ops.conv2d.Conv2D('Conv1', 1, 16, 3, output))
    output = tf.nn.relu(lib.ops.conv2d.Conv2D('Conv2', 16, 32, 3, output, stride=2))
    output = tf.nn.relu(lib.ops.conv2d.Conv2D('Conv3', 32, 32, 3, output, stride=2))

    # TODO if we actually cared about making a good autoencoder, put a bottleneck
    # layer here

    output = tf.nn.relu(lib.ops.deconv2d.Deconv2D('Deconv2', 32, 32, 3, output))
    output = tf.nn.relu(lib.ops.deconv2d.Deconv2D('Deconv3', 32, 16, 3, output))
    output = lib.ops.conv2d.Conv2D('Output', 16, 1, 3, output)

    output = tf.reshape(output, [-1, 784])

    cost = tf.reduce_mean(tf.square(output - inputs))

    train_data, dev_data, test_data = lib.mnist.load(
        BATCH_SIZE,
        BATCH_SIZE
    )

    lib.train_loop.train_loop(
        session=session,
        inputs=[inputs, targets],
        cost=cost,
        optimizer=tf.train.AdamOptimizer(LR),
        train_data=train_data,
        test_data=dev_data,
        times=TIMES
    )