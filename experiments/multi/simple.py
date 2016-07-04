# from theano import function, config, shared, tensor, sandbox
# import numpy
# import time

# vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
# iters = 1000

# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], tensor.exp(x))
# print(f.maker.fgraph.toposort())
# t0 = time.time()
# for i in range(iters):
#     r = f()
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
# if numpy.any([isinstance(x.op, tensor.Elemwise) and
#               ('Gpu' not in type(x.op).__name__)
#               for x in f.maker.fgraph.toposort()]):
#     print('Used the cpu')
# else:
#     print('Used the gpu')


# v01 = v01.transfer('dev1')
# v02 = v02.transfer('dev1')
# v11 = v11.transfer('dev1')
# v12 = v12.transfer('dev1')

import numpy
import theano
import time
import theano.tensor as T

DIM = 8192

v01 = theano.shared(numpy.random.random((DIM,DIM)).astype('float32'), target='dev0')
v02 = theano.shared(numpy.random.random((DIM,DIM)).astype('float32'), target='dev0')
v11 = theano.shared(numpy.random.random((DIM,DIM)).astype('float32'), target='dev1')
v12 = theano.shared(numpy.random.random((DIM,DIM)).astype('float32'), target='dev1')

result1 = v02
result2 = v12
for i in xrange(10):
    result1 = theano.tensor.dot(v01, result1)
    result2 = theano.tensor.dot(v11, result2)

result = T.mean(result1 + result2)

f = theano.function([], [result])

f() # call once for warm-up

t0 = time.time()
f()
print time.time() - t0
