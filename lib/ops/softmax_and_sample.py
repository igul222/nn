import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams(seed=234)

def softmax_and_sample(logits):
    old_shape = logits.shape
    flattened_logits = logits.reshape((-1, logits.shape[logits.ndim-1]))
    samples = T.cast(
        srng.multinomial(pvals=T.nnet.softmax(flattened_logits)),
        theano.config.floatX
    ).reshape(old_shape)
    return T.argmax(samples, axis=samples.ndim-1)