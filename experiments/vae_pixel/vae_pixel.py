"""
VAE + Pixel RNN/CNN
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import lsun

import lib
import lib.debug
import lib.mnist_binarized
import lib.mnist_256ary
import lib.train_loop
import lib.ops.mlp
import lib.ops.conv_encoder
import lib.ops.conv_decoder
import lib.ops.kl_unit_gaussian
import lib.ops.conv2d
import lib.ops.diagonal_bilstm
import lib.ops.relu
import lib.ops.softmax_nll
import lib.ops.softmax_and_sample
import lib.ops.embedding

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
from scipy.misc import imsave
import lasagne

import time
import functools

MODE = '256ary' # binary or 256ary

MODEL = 'pixel_cnn' # either pixel_cnn or pixel_rnn
PIX_DIM = 128
PIXEL_CNN_LAYERS = 4
PIXEL_CNN_FILTER_SIZE = 3
SKIP_PIX_CNN = True

CONV_BASE_N_FILTERS = 32
CONV_N_POOLS = 3
CONV_FILTER_SIZE = 3
CONV_BN = False # never seemed to help
CONV_DEEP = False

LATENT_DIM = 128
ALPHA_ITERS = 10000
VANILLA = False
LR = 1e-4

NEW_ANNEAL = False

DATASET = 'mnist'
LSUN_DOWNSAMPLE = True

TIMES = ('iters', 1000, 200000, 1000)

if DATASET == 'lsun':
    if LSUN_DOWNSAMPLE:
        CONV_N_POOLS = 3
        TIMES = ('iters', 2000, 400*1000, 20000) # 32dim
    else:
        CONV_N_POOLS = 4
        TIMES = ('iters', 2000, 400*1000, 40000) # 64dim

    BATCH_SIZE = 64
    N_CHANNELS = 3
    if LSUN_DOWNSAMPLE:
        HEIGHT = 32
        WIDTH = 32
    else:
        HEIGHT = 64
        WIDTH = 64
    train_data, dev_data = lsun.load(BATCH_SIZE, LSUN_DOWNSAMPLE)
else:
    BATCH_SIZE = 100
    N_CHANNELS = 1
    HEIGHT = 28
    WIDTH = 28
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
        embedded = embedded.dimshuffle(0,4,1,2,3)
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
            inputs=embedded,
            batchnorm=CONV_BN,
            deep=CONV_DEEP
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
            inputs=inputs,
            batchnorm=CONV_BN,
            deep=CONV_DEEP
        )

    return mu_and_log_sigma[:, ::2], mu_and_log_sigma[:, 1::2]

def Decoder(latents, images):
    # We apply the sigmoid at a later step

    decoder_out = lib.ops.conv_decoder.ConvDecoder(
        'Decoder',
        input_dim=LATENT_DIM,
        n_unpools=CONV_N_POOLS,
        base_n_filters=CONV_BASE_N_FILTERS,
        filter_size=CONV_FILTER_SIZE,
        output_size=WIDTH,
        output_n_channels=PIX_DIM,
        inputs=latents,
        batchnorm=CONV_BN,
        deep=CONV_DEEP
    )

    if SKIP_PIX_CNN:

        output = lib.ops.conv2d.Conv2D(
            'OutputConv',
            input_dim=PIX_DIM,
            output_dim=256*N_CHANNELS,
            filter_size=1,
            inputs=decoder_out,
            he_init=False
        ).reshape((-1, 256, N_CHANNELS, HEIGHT, WIDTH)).dimshuffle(0,2,3,4,1)

        return output

    else:


        if MODE=='256ary':

            embedded = lib.ops.embedding.Embedding(
                'Decoder.Embedding', 
                256, 
                PIX_DIM, 
                images
            )
            embedded = embedded.dimshuffle(0,4,1,2,3)
            embedded = embedded.reshape((
                embedded.shape[0], 
                embedded.shape[1] * embedded.shape[2], 
                embedded.shape[3], 
                embedded.shape[4]
            ))

            output = lib.ops.conv2d.Conv2D(
                'InputConv', 
                input_dim=N_CHANNELS*PIX_DIM, 
                output_dim=PIX_DIM, 
                filter_size=7, 
                inputs=embedded, 
                mask_type=('a', N_CHANNELS),
                he_init=False
            )

        else:

            output = lib.ops.conv2d.Conv2D(
                'InputConv', 
                input_dim=N_CHANNELS, 
                output_dim=PIX_DIM, 
                filter_size=7, 
                inputs=images, 
                mask_type=('a', N_CHANNELS),
                he_init=False
            )

        output = T.concatenate([output, decoder_out], axis=1)

        if MODEL=='pixel_rnn':

            output = lib.ops.diagonal_bilstm.DiagonalBiLSTM(
                'DiagonalBiLSTM', 
                input_dim=2*PIX_DIM, 
                output_dim=PIX_DIM, 
                input_shape=(N_CHANNELS, HEIGHT, WIDTH),
                inputs=output
            )

        elif MODEL=='pixel_cnn':

            for i in xrange(PIXEL_CNN_LAYERS):
                if i==0:
                    inp_dim = 2*PIX_DIM
                else:
                    inp_dim = PIX_DIM

                output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(
                    'PixelCNNConv'+str(i),
                    input_dim=inp_dim,
                    output_dim=PIX_DIM,
                    filter_size=PIXEL_CNN_FILTER_SIZE,
                    inputs=output,
                    mask_type=('b', N_CHANNELS),
                ))

        output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(
            'OutputConv1', 
            input_dim=PIX_DIM, 
            output_dim=PIX_DIM, 
            filter_size=1, 
            inputs=output, 
            mask_type=('b', N_CHANNELS),
        ))

        output = lib.ops.relu.relu(lib.ops.conv2d.Conv2D(
            'OutputConv2', 
            input_dim=PIX_DIM, 
            output_dim=PIX_DIM, 
            filter_size=1, 
            inputs=output, 
            mask_type=('b', N_CHANNELS), 
        ))

        if MODE=='256ary':
            output = lib.ops.conv2d.Conv2D(
                'OutputConv3',
                input_dim=PIX_DIM,
                output_dim=256*N_CHANNELS,
                filter_size=1,
                inputs=output,
                mask_type=('b', N_CHANNELS),
                he_init=False
            ).reshape((-1, 256, N_CHANNELS, HEIGHT, WIDTH)).dimshuffle(0,2,3,4,1)
        else:
            output = lib.ops.conv2d.Conv2D(
                'OutputConv3',
                input_dim=PIX_DIM,
                output_dim=N_CHANNELS,
                filter_size=1,
                inputs=output,
                mask_type=('b', N_CHANNELS),
                he_init=False
            )

        return output

total_iters = T.iscalar('total_iters')
images = T.itensor4('images') # shape: (batch size, n channels, height, width)

mu, log_sigma = Encoder(images)

# mu = lib.debug.print_stats('mu', mu)
# log_sigma = lib.debug.print_stats('log_sigma', log_sigma)

if VANILLA:
    latents = mu
else:
    eps = T.cast(theano_srng.normal(mu.shape), theano.config.floatX)
    latents = mu + (eps * T.exp(log_sigma))

latents = T.minimum(50, latents)
latents = T.maximum(-50, latents)

output = Decoder(latents, images)

if MODE=='256ary':
    # reconst_cost = lib.ops.softmax_nll.softmax_nll(output, images).mean()

    reconst_cost = T.nnet.categorical_crossentropy(
        T.nnet.softmax(output.reshape((-1, 256))),
        images.flatten()
    )
    # # reconst_cost = lib.debug.print_stats('reconst_cost', reconst_cost)
    reconst_cost = reconst_cost.mean()
    # # reconst_cost = lib.debug.print_stats('mean_reconst_cost', reconst_cost)

else:
    reconst_cost = T.nnet.binary_crossentropy(
        T.nnet.sigmoid(output), 
        inputs
    ).sum() / inputs.shape[0].astype(theano.config.floatX)


reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(mu, log_sigma).mean()
reg_cost *= lib.floatX(float(LATENT_DIM) / (WIDTH*HEIGHT*N_CHANNELS))

alpha = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS)
)

if VANILLA:
    cost = reconst_cost
else:
    if NEW_ANNEAL:
        capped_reg_cost = T.maximum(
            2*(1-alpha),
            reg_cost
        )
        cost = reconst_cost + capped_reg_cost
    else:
        cost = reconst_cost + (alpha * reg_cost)
sample_fn_latents = T.matrix('sample_fn_latents')
sample_fn_output = Decoder(sample_fn_latents, images)
if MODE=='256ary':
    sample_fn = theano.function(
        [sample_fn_latents, images],
        sample_fn_output,
        on_unused_input='warn'
    )
else:
    sample_fn = theano.function(
        [sample_fn_latents, images],
        T.nnet.sigmoid(sample_fn_output),
        on_unused_input='warn'
    )

sample_fn_latents = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
sample_fn_latents = sample_fn_latents.astype(theano.config.floatX)
def generate_and_save_samples(tag):
    def save_images(images, filename):
        """images.shape: (batch, n channels, height, width)"""
        images = images.reshape((10,10,28,28))
        # rowx, rowy, height, width -> rowy, height, rowx, width
        images = images.transpose(1,2,0,3)
        images = images.reshape((10*28, 10*28))

        image = scipy.misc.toimage(images, cmin=0.0, cmax=1.0)
        image.save('{}_{}.jpg'.format(filename, tag))

    def color_grid_vis(X, nh, nw, save_path):
        # from github.com/Newmu
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
        for n, x in enumerate(X):
            j = n/nw
            i = n%nw
            img[j*h:j*h+h, i*w:i*w+w, :] = x
        imsave(save_path, img)

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

    # last_sample_fn_hash = np.random.normal(
    #     size=(N_CHANNELS, HEIGHT, WIDTH),
    # ).astype('float32')

    import sys

    for j in xrange(HEIGHT):
        for k in xrange(WIDTH):
            for i in xrange(N_CHANNELS):
                next_sample = sample_fn(sample_fn_latents, samples)

                # For debugging:
                # sample_fn_hash = next_sample[0,:,:,:,0]
                # diff = (sample_fn_hash != last_sample_fn_hash)
                # for h in xrange(HEIGHT):
                #     for w in xrange(WIDTH):
                #         for ch in xrange(N_CHANNELS):
                #             if diff[ch,h,w]:
                #                 sys.stdout.write('X')
                #             else:
                #                 sys.stdout.write('.')
                #         sys.stdout.write(' ')
                #     sys.stdout.write("\n")

                # last_sample_fn_hash = sample_fn_hash

                if MODE=='binary':
                    next_sample = binarize(next_sample)
                    samples[:, i, j, k] = next_sample[:, i, j, k]
                else:
                    pre_softmax = next_sample[:, i, j, k]
                    pre_softmax = np.array(pre_softmax, dtype='float64')
                    shift = np.max(pre_softmax, axis=pre_softmax.ndim-1, keepdims=True)
                    exp = np.exp(pre_softmax - shift)
                    softmax = exp / np.sum(exp, axis=exp.ndim-1, keepdims=True)
                    for h in xrange(softmax.shape[0]):
                        next_sample = np.argmax(np.random.multinomial(1, softmax[h], size=1))
                        samples[h, i, j, k] = next_sample

    if DATASET=='lsun':
        sqrt_n_samples = int(np.sqrt(BATCH_SIZE))
        color_grid_vis(
            samples, 
            sqrt_n_samples, 
            sqrt_n_samples, 
            'samples_{}.png'.format(tag)
        )
    else:
        if MODE=='256ary':
            samples = samples / 255.
        save_images(samples, 'samples')

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