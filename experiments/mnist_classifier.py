"""
Conv VAE
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

sys.setrecursionlimit(2000)

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import lib
import lib.train_loop
import lib.mnist
import lib.ops.mlp

import numpy as np
import theano
import theano.tensor as T
import lasagne

import functools

LR = 1e-3
BATCH_SIZE = 100

TIMES = ('iters', 1*500, 10*500)
# TIMES = ('seconds', 60*30, 60*60*6)

lib.print_model_settings(locals().copy())

inputs = T.matrix('inputs')
targets = T.ivector('targets')

output = T.nnet.softmax(
    lib.ops.mlp.MLP(
        'MLP',
        input_dim=784,
        hidden_dim=512,
        output_dim=10,
        n_layers=20,
        inputs=inputs
    )
)

cost = T.nnet.categorical_crossentropy(
    output,
    targets
).mean()

acc = T.eq(T.argmax(output, axis=1), targets).mean()

train_data, dev_data, test_data = lib.mnist.load(
    BATCH_SIZE, 
    BATCH_SIZE
)

lib.train_loop.train_loop(
    inputs=[inputs, targets],
    cost=cost,
    prints=[
        ('acc', acc)
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    test_data=dev_data,
    times=TIMES
)