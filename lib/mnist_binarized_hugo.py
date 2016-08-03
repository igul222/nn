import numpy as np

MNIST_HUGO_PATH = '/home/ishaan/mnist_binarized_hugo/'

def mnist_generator(images, batch_size):
    def get_epoch():
        np.random.shuffle(images)
        image_batches = images.reshape(-1, batch_size, 1, 28, 28)
        for i in xrange(len(image_batches)):
            yield (np.copy(image_batches[i]),)

    return get_epoch

def load(batch_size):
    datasets = []
    for split in ['train', 'valid', 'test']:
        with open(MNIST_HUGO_PATH+'binarized_mnist_'+split+'.amat', 'r') as f:
            data = []
            for line in f:
                data.append([float(x) for x in line.split()])
            datasets.append(np.array(data, dtype='float32'))

    return (
        mnist_generator(datasets[0], batch_size), 
        mnist_generator(datasets[1], batch_size), 
        mnist_generator(datasets[2], batch_size)
    )
