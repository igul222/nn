import numpy as np
import scipy.misc
import os
import time

DATA_DIR = '/home/ubuntu/lsun/bedrooms/'

def load(batch_size):
    def generator():
        with open(DATA_DIR+'files.txt', 'r') as f:
            files = [l[:-1] for l in f]
        images = np.zeros((batch_size, 3, 256, 256), dtype='int32')
        random_state = np.random.RandomState(42)
        random_state.shuffle(files)
        for i, path in enumerate(files):
            image = scipy.misc.imread(
                os.path.normpath(os.path.join(DATA_DIR, path))
            )
            image = image.transpose(2,0,1)
            offset_y = (image.shape[1]-256)/2
            offset_x = (image.shape[2]-256)/2
            images[i % batch_size] = image[:, offset_y:offset_y+256, offset_x:offset_x+256]
            if i > 0 and i % batch_size == 0:
                yield (images,)
    return generator

if __name__ == '__main__':
    train_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()