import numpy as np
import theano
import theano.tensor as T

import time

w = theano.shared(np.random.normal(size=(8192,8192)).astype('float32'))

w1 = theano.shared(np.random.normal(size=(8192,128)).astype('float32'))
w2 = theano.shared(np.random.normal(size=(128,8192)).astype('float32'))

x = theano.shared(np.random.normal(size=(128,256,8192)).astype('float32'))

# f = theano.function([], T.dot(x,w)[0,0,0])
f = theano.function([], T.dot(T.dot(x,w1), w2)[0,0,0])

# f1 = theano.function([], T.dot(x,w1)[0,0,0])

# f()
# f1()

# t0=time.time()
# f1()
# print time.time() - t0

t0=time.time()
f()
print time.time() - t0