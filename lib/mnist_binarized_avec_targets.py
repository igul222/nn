import lib.mnist

import numpy as np
import theano

def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (np.random.uniform(size=images.shape) < images).astype(theano.config.floatX)


def binarized_generator(generator):
    def get_epoch():
        for images, targets in generator():
            images = images.reshape((-1, 1, 28, 28))
            images = binarize(images)
            yield (images,targets)
    return get_epoch

def load(batch_size, test_batch_size):
    train_gen, dev_gen, test_gen = lib.mnist.load(batch_size, test_batch_size)
    return (
        binarized_generator(train_gen),
        binarized_generator(dev_gen),
        binarized_generator(test_gen)
    )
