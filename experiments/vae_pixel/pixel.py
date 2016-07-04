"""
Pixel RNN on MNIST
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import lib
import lib.train_loop
import lib.mnist_binarized
import lib.mnist_256ary
import lib.ops.conv2d
import lib.ops.diagonal_bilstm
import lib.ops.relu
import lib.ops.softmax_and_sample
import lib.ops.embedding

import numpy as np
import theano
import theano.tensor as T
import scipy.misc
import lasagne

import functools

MODE = '256ary' # binary or 256ary

MODEL = 'pixel_cnn' # either pixel_cnn or pixel_rnn
DIM = 32
PIXEL_CNN_LAYERS = 4

LR = 2e-4

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

TIMES = ('iters', 10*500, 1000*500)

lib.print_model_settings(locals().copy())

# inputs.shape: (batch size, n channels, height, width)
if MODE=='256ary':
    inputs = T.itensor4('inputs')
    inputs_embed = lib.ops.embedding.Embedding('Embedding', 256, DIM, inputs)
    inputs_embed = inputs_embed.dimshuffle(0,1,4,2,3)
    inputs_embed = inputs_embed.reshape((inputs_embed.shape[0], inputs_embed.shape[1] * inputs_embed.shape[2], inputs_embed.shape[3], inputs_embed.shape[4]))

    output = lib.ops.conv2d.Conv2D(
        'InputConv', 
        input_dim=N_CHANNELS*DIM, 
        output_dim=DIM, 
        filter_size=7, 
        inputs=inputs_embed, 
        mask_type=('a', N_CHANNELS),
        he_init=False
    )
else:
    inputs = T.tensor4('inputs')
    inputs_float = inputs

    output = lib.ops.conv2d.Conv2D(
        'InputConv', 
        input_dim=N_CHANNELS, 
        output_dim=DIM, 
        filter_size=7, 
        inputs=inputs_float, 
        mask_type=('a', N_CHANNELS),
        he_init=False
    )

if MODEL=='pixel_rnn':

    output = lib.ops.diagonal_bilstm.DiagonalBiLSTM(
        'DiagonalBiLSTM', 
        input_dim=DIM, 
        output_dim=DIM,
        input_shape=(N_CHANNELS, HEIGHT, WIDTH),
        inputs=output
    )

elif MODEL=='pixel_cnn':

    for i in xrange(PIXEL_CNN_LAYERS):
        output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(
            'PixelCNNConv'+str(i),
            input_dim=DIM,
            output_dim=DIM,
            filter_size=3,
            inputs=output,
            mask_type=('b', N_CHANNELS),
        ))

output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(
    'OutputConv1', 
    input_dim=DIM, 
    output_dim=DIM, 
    filter_size=1, 
    inputs=output, 
    mask_type=('b', N_CHANNELS), 
))

output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(
    'OutputConv2', 
    input_dim=DIM, 
    output_dim=DIM, 
    filter_size=1, 
    inputs=output, 
    mask_type=('b', N_CHANNELS), 
))

if MODE=='256ary':
    output = lib.ops.conv2d.Conv2D(
        'OutputConv3',
        input_dim=DIM,
        output_dim=256*N_CHANNELS,
        filter_size=1,
        inputs=output,
        mask_type=('b', N_CHANNELS),
        he_init=False
    ).reshape((-1, 256, N_CHANNELS, HEIGHT, WIDTH)).dimshuffle(0,2,3,4,1)
else:
    output = lib.ops.conv2d.Conv2D(
        'OutputConv3',
        input_dim=DIM,
        output_dim=N_CHANNELS,
        filter_size=1,
        inputs=output,
        mask_type=('b', N_CHANNELS),
        he_init=False
    )

if MODE=='256ary':
    cost = T.nnet.categorical_crossentropy(
        T.nnet.softmax(output.reshape((-1,256))),
        inputs.flatten()
    ).mean()

    sample_fn = theano.function(
        [inputs],
        lib.ops.softmax_and_sample.softmax_and_sample(output)
    )
else:
    cost = T.nnet.binary_crossentropy(
        T.nnet.sigmoid(output), 
        inputs
    ).sum() / inputs.shape[0].astype(theano.config.floatX)

    sample_fn = theano.function(
        [inputs],
        T.nnet.sigmoid(output)
    )

def generate_and_save_samples(tag):

    def save_images(images, filename):
        """images.shape: (batch, n channels, height, width)"""
        images = images.reshape((10,10,28,28))
        # rowx, rowy, height, width -> rowy, height, rowx, width
        images = images.transpose(1,2,0,3)
        images = images.reshape((10*28, 10*28))

        image = scipy.misc.toimage(images, cmin=0.0, cmax=1.0)
        image.save('{}_{}.jpg'.format(filename, tag))

    def binarize(images):
        """
        Stochastically binarize values in [0, 1] by treating them as p-values of
        a Bernoulli distribution.
        """
        return (
            np.random.uniform(size=images.shape) < images
        ).astype(theano.config.floatX)

    if MODE=='256ary':
        dtype = 'int32'
    else:
        dtype = theano.config.floatX
    samples = np.zeros(
        (BATCH_SIZE, N_CHANNELS, HEIGHT, WIDTH), 
        dtype=dtype
    )

    for j in xrange(HEIGHT):
        for k in xrange(WIDTH):
            for i in xrange(N_CHANNELS):
                next_sample = sample_fn(samples)
                if MODE=='binary':
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]

    if MODE=='256ary':
        samples = samples / 255.

    save_images(samples, 'samples')


if MODE=='256ary':
    train_data, dev_data, test_data = lib.mnist_256ary.load(
        BATCH_SIZE, 
        BATCH_SIZE
    )
else:
    train_data, dev_data, test_data = lib.mnist_binarized.load(
        BATCH_SIZE, 
        BATCH_SIZE
    )

lib.train_loop.train_loop(
    inputs=[inputs],
    cost=cost,
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    test_data=dev_data,
    callback=generate_and_save_samples,
    times=TIMES
)