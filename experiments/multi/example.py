import os, sys
sys.path.append(os.getcwd())

# try: # This only matters on Ishaan's computer
#     import experiment_tools
#     experiment_tools.wait_for_gpu(high_priority=False)
# except ImportError:
#     pass

import theano_multi

import numpy as np
import theano
import theano.tensor as T

import time

BATCH_SIZE = 512
DIM = 4096

x = T.matrix('x')
y = T.matrix('y')

W = theano.shared(
    np.random.normal(scale=0.01, size=(DIM, DIM)).astype(theano.config.floatX)
)

y_hat = x
for i in xrange(50):
    y_hat = T.dot(y_hat, W)

cost = T.mean((y_hat - y)**2)

params = [W]

grads = T.grad(cost, params)

# grads = theano_multi.multi(grads, params=params, other_contexts=['dev2', 'dev1'])

updates = [(p, p - 0.1*g) for p, g in zip(params, grads)]

# diffs = [T.sum((pg - g)**2) for pg,g in zip(p_grads, grads)]

train_fn = theano.function(
    [x,y],
    cost,
    updates=updates,
    on_unused_input='warn'
)

for i in xrange(100):
    
    x = np.random.normal(size=(BATCH_SIZE, DIM)).astype(theano.config.floatX)
    y = x

    t0 = time.time()
    cost = train_fn(x,y)
    print "cost:{}\ttime:{}".format(
        cost,
        time.time() - t0
    )