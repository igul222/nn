import tflib.mnist

import numpy as np

def discretize(x):
    return (x*(256-1e-8)).astype('int32')

def binarized_generator(generator, include_targets=False):
    def get_epoch():
        for images, targets in generator():
            images = images.reshape((-1, 1, 28, 28))
            images = discretize(images)
            if include_targets:
                yield (images, targets)
            else:
                yield (images,)
    return get_epoch

def load(batch_size, test_batch_size):
    train_gen, dev_gen, test_gen = tflib.mnist.load(batch_size, test_batch_size)
    return (
        binarized_generator(train_gen),
        binarized_generator(dev_gen),
        binarized_generator(test_gen, True)
    )