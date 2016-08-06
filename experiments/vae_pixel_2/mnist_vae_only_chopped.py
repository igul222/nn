"""
VAE + Pixel CNN
Ishaan Gulrajani
"""

import os, sys, time
sys.path.append(os.getcwd())
sys.path.append('/u/ahmedfar/Tmp/nn/')

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import lib
import lib.train_loop
import lib.mnist_binarized
import lib.mnist_binarized_fuel
import lib.ops.mlp
import lib.ops.conv_encoder
import lib.ops.conv_decoder
import lib.ops.kl_unit_gaussian
import lib.ops.conv2d
import lib.ops.deconv2d
import lib.ops.diagonal_bilstm
import lib.ops.relu

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
import lasagne

import functools

theano.config.dnn.conv.algo_fwd = 'time_on_shape_change'
theano.config.dnn.conv.algo_bwd_filter = 'time_on_shape_change'
theano.config.dnn.conv.algo_bwd_data = 'time_on_shape_change'

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

HUGO_TRAIN = True
HUGO_TEST = True

DIM_1 = 32
DIM_2 = 32
DIM_3 = 64
DIM_4 = 64
DIM_PIX = 32
PIXEL_CNN_FILTER_SIZE = 5
PIXEL_CNN_LAYERS = 6

LATENT_DIM = 64
ALPHA_ITERS = 10000
VANILLA = False
LR = 1e-3

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

EXPNAME = '/data/lisatmp4/faruk/pixelvae/vae_only' + str(time.time()) + '/'

TEST_BATCH_SIZE = 100
TIMES = ('iters', 1*500, 600*500, 1*500, 100*500, 100*500)

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def PixCNNGate(x):
    a = x[:,::2]
    b = x[:,1::2]
    return T.tanh(a) * T.nnet.sigmoid(b)

def Encoder(inputs):

    output = inputs

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.2', input_dim=DIM_1,      output_dim=DIM_2, filter_size=3, inputs=output, stride=2))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.3', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.4', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, inputs=output, stride=2))

    # Pad from 7x7 to 8x8
    padded = T.zeros((output.shape[0], output.shape[1], 8, 8), dtype='float32')
    output = T.inc_subtensor(padded[:,:,:7,:7], output)

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.5', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.6', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, inputs=output, stride=2))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.7', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Enc.8', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = output.reshape((output.shape[0], -1))
    output = lib.ops.linear.Linear('Enc.Out', input_dim=4*4*DIM_4, output_dim=2*LATENT_DIM, inputs=output)
    return output[:, ::2], output[:, 1::2]

def Decoder(latents):

    output = latents

    output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
    output = output.reshape((output.shape[0], DIM_4, 4, 4))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

    # Cut from 8x8 to 7x7
    output = output[:,:,:7,:7]

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

    output = T.nnet.relu(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

    output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_1, output_dim=N_CHANNELS, filter_size=1, inputs=output, he_init=False)

    return output

if not os.path.exists(EXPNAME):  os.makedirs(EXPNAME)

total_iters = T.iscalar('total_iters')
images = T.tensor4('images') # shape: (batch size, n channels, height, width)

mu, log_sigma = Encoder(images)

if VANILLA:
    latents = mu
else:
    eps = T.cast(theano_srng.normal(mu.shape), theano.config.floatX)
    latents = mu + (eps * T.exp(log_sigma))

# Theano bug: NaNs unless I pass 2D tensors to binary_crossentropy
reconst_cost = T.nnet.binary_crossentropy(
    T.nnet.sigmoid(
        Decoder(latents).reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
    ),
    images.reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
).mean(axis=0).sum()

reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(
    mu, 
    log_sigma
).mean(axis=0).sum()

alpha = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS)
)

if VANILLA:
    cost = reconst_cost
else:
    cost = reconst_cost + (alpha * reg_cost)

sample_fn_latents = T.matrix('sample_fn_latents')
sample_fn = theano.function(
    [sample_fn_latents],
    T.nnet.sigmoid(Decoder(sample_fn_latents))
)

eval_fn = theano.function(
    [images, total_iters],
    cost
)

train_data, dev_data, test_data = lib.mnist_binarized.load(
    BATCH_SIZE, 
    TEST_BATCH_SIZE
)

fuel_train_data, fuel_dev_data, fuel_test_data = lib.mnist_binarized_fuel.load(
    BATCH_SIZE
)
if HUGO_TRAIN:
    train_data = fuel_train_data
if HUGO_TEST:
    dev_data = fuel_dev_data
    test_data = fuel_test_data

def generate_and_save_samples(tag):

    costs = []
    for (images,) in test_data():
        costs.append(eval_fn(images, ALPHA_ITERS+1))
    print "test cost: {}".format(np.mean(costs))

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

    latents = np.random.normal(size=(100, LATENT_DIM))
    latents = latents.astype(theano.config.floatX)

    samples = np.zeros(
        (100, N_CHANNELS, HEIGHT, WIDTH), 
        dtype=theano.config.floatX
    )

    for j in xrange(HEIGHT):
        for k in xrange(WIDTH):
            for i in xrange(N_CHANNELS):
                next_sample = binarize(sample_fn(latents))
                samples[:, i, j, k] = next_sample[:, i, j, k]

    save_images(samples, EXPNAME + '/sample')

lib.train_loop.train_loop(
    inputs=[total_iters, images],
    inject_total_iters=True,
    cost=cost,
    prints=[
        ('alpha', alpha), 
        ('reconst', reconst_cost), 
        ('reg', reg_cost)
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    test_data=dev_data,
    callback=generate_and_save_samples,
    times=TIMES
)
