"""
Conv VAE
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=True)
except ImportError:
    pass

import lib
import lib.train_loop
import lib.mnist_binarized
import lib.mnist_256ary
import lib.ops.mlp
import lib.ops.conv_encoder
import lib.ops.conv_decoder
import lib.ops.kl_unit_gaussian
import lib.ops.softmax_and_sample
import lib.ops.embedding

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
import lasagne

import functools

MODE = '256ary' # binary or 256ary

CONV_BASE_N_FILTERS = 16
CONV_N_POOLS = 3
CONV_FILTER_SIZE = 3

LATENT_DIM = 128
ALPHA_ITERS = 20000
VANILLA = False
LR = 2e-4

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

TIMES = ('iters', 10*500, 1000*500)
# TIMES = ('seconds', 60*30, 60*60*6)

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def Encoder(inputs):
    if MODE=='256ary':
        embedded = lib.ops.embedding.Embedding(
            'Encoder.Embedding', 
            256, 
            CONV_BASE_N_FILTERS, 
            inputs
        )
        embedded = embedded.dimshuffle(0,1,4,2,3)
        embedded = embedded.reshape((
            embedded.shape[0], 
            embedded.shape[1] * embedded.shape[2], 
            embedded.shape[3], 
            embedded.shape[4]
        ))

        mu_and_log_sigma = lib.ops.conv_encoder.ConvEncoder(
            'Encoder',
            input_n_channels=N_CHANNELS*CONV_BASE_N_FILTERS,
            input_size=WIDTH,
            n_pools=CONV_N_POOLS,
            base_n_filters=CONV_BASE_N_FILTERS,
            filter_size=CONV_FILTER_SIZE,
            output_dim=2*LATENT_DIM,
            inputs=embedded
        )
    else:
        mu_and_log_sigma = lib.ops.conv_encoder.ConvEncoder(
            'Encoder',
            input_n_channels=N_CHANNELS,
            input_size=WIDTH,
            n_pools=CONV_N_POOLS,
            base_n_filters=CONV_BASE_N_FILTERS,
            filter_size=CONV_FILTER_SIZE,
            output_dim=2*LATENT_DIM,
            inputs=inputs
        )

    return mu_and_log_sigma[:, ::2], mu_and_log_sigma[:, 1::2]

def Decoder(latents):
    # We apply the sigmoid at a later step
    if MODE=='256ary':
        return lib.ops.conv_decoder.ConvDecoder(
            'Decoder',
            input_dim=LATENT_DIM,
            n_unpools=CONV_N_POOLS,
            base_n_filters=CONV_BASE_N_FILTERS,
            filter_size=CONV_FILTER_SIZE,
            output_size=WIDTH,
            output_n_channels=256*N_CHANNELS,
            inputs=latents
        ).reshape(
            (-1, 256, N_CHANNELS, HEIGHT, WIDTH)
        ).dimshuffle(0,2,3,4,1)
    else:
        return lib.ops.conv_decoder.ConvDecoder(
            'Decoder',
            input_dim=LATENT_DIM,
            n_unpools=CONV_N_POOLS,
            base_n_filters=CONV_BASE_N_FILTERS,
            filter_size=CONV_FILTER_SIZE,
            output_size=WIDTH,
            output_n_channels=N_CHANNELS,
            inputs=latents
        )


total_iters = T.iscalar('total_iters')
if MODE=='256ary':
    images = T.itensor4('images')
else:
    images = T.tensor4('images') # shape (batch size, n channels, height, width)

mu, log_sigma = Encoder(images)

if VANILLA:
    latents = mu
else:
    eps = T.cast(theano_srng.normal(mu.shape), theano.config.floatX)
    latents = mu + (eps * T.exp(log_sigma))

outputs = Decoder(latents)

if MODE=='256ary':
    reconst_cost = T.nnet.categorical_crossentropy(
        T.nnet.softmax(outputs.reshape((-1, 256))),
        images.flatten()
    ).mean()
else:
    # Theano bug: NaNs unless I pass 2D tensors to binary_crossentropy.
    reconst_cost = T.nnet.binary_crossentropy(
        T.nnet.sigmoid(outputs.reshape((-1, N_CHANNELS*HEIGHT*WIDTH))),
        images.reshape((-1, N_CHANNELS*HEIGHT*WIDTH))
    ).mean(axis=0).sum()

reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(mu, log_sigma).sum()
reg_cost /= lib.floatX(WIDTH*HEIGHT*N_CHANNELS*BATCH_SIZE)

alpha = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS)
)

if VANILLA:
    cost = reconst_cost
else:
    cost = reconst_cost + (alpha * reg_cost)

rand_z = T.cast(theano_srng.normal((100, LATENT_DIM)), theano.config.floatX)
sample_fn_output = Decoder(rand_z)
if MODE=='256ary':
    sample_fn = theano.function(
        [],
        lib.ops.softmax_and_sample.softmax_and_sample(sample_fn_output)
    )
else:
    sample_fn = theano.function(
        [],
        T.nnet.sigmoid(sample_fn_output)
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
        save_images(sample_fn() / 255., 'samples')
    else:
        save_images(binarize(sample_fn()), 'samples')

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