import os, sys
sys.path.append(os.getcwd())

import numpy as np

import lib
import lib.audio

import functools

DATA_PATH = '/media/seagate/blizzard/parts'
N_FILES = 141703

BATCH_SIZE = 512
SEQ_LEN = 128
Q_LEVELS = 256

Q_ZERO = np.int32(Q_LEVELS//2)

train_data = functools.partial(
    lib.audio.feed_epoch,
    DATA_PATH,
    N_FILES,
    BATCH_SIZE,
    SEQ_LEN,
    0,
    Q_LEVELS,
    Q_ZERO
)

for i, (minibatch, reset) in enumerate(train_data()):
    o_minibatch = minibatch
    minibatch = minibatch.astype('int32')
    if np.min(minibatch) < 0 or np.max(minibatch) > 255:
        print "problem! min {} max {}".format(
            np.min(minibatch),
            np.max(minibatch)
        )
        print "problem! min {} max {}".format(
            np.min(o_minibatch),
            np.max(o_minibatch)
        )
        raise Exception()        
    print i

print "Done!"