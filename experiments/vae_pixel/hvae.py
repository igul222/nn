"""
Conv VAE
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
import lib.debug
import lib.train_loop
import lib.mnist_binarized
import lib.mnist_256ary
import lib.ops.mlp
import lib.ops.conv2d
import lib.ops.deconv2d
import lib.ops.conv_encoder
import lib.ops.conv_decoder
import lib.ops.kl_unit_gaussian
import lib.ops.kl_gaussian_gaussian
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

# CONV_BASE_N_FILTERS = 16
# CONV_N_POOLS = 3
CONV_FILTER_SIZE = 3

CONV_DIM = 64
L2_CONV_DIM = 128
L3_CONV_DIM = 256
L4_CONV_DIM = 512
L5_FC_DIM = 2048

LATENT_DIM = 64
ALPHA_ITERS = 20000
VANILLA = False
LR = 2e-4

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

TIMES = ('iters', 1000, 500000, 1000)
# TIMES = ('seconds', 60*30, 60*60*6, 60*30)

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def Encoder(name, input_dim, hidden_dim, latent_dim, downsample, inputs):

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Input',
        input_dim=input_dim,
        output_dim=hidden_dim,
        filter_size=1,
        inputs=inputs
    ))

    if downsample:
        output = T.nnet.relu(lib.ops.conv2d.Conv2D(
            name+'.Down',
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            filter_size=CONV_FILTER_SIZE,
            inputs=output,
            stride=2
        ))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Conv1',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=CONV_FILTER_SIZE,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Conv2',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=CONV_FILTER_SIZE,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Conv3',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=CONV_FILTER_SIZE,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Conv4',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=CONV_FILTER_SIZE,
        inputs=output,
    ))

    output = lib.ops.conv2d.Conv2D(
        name+'.Output',
        input_dim=hidden_dim,
        output_dim=2*latent_dim,
        filter_size=1,
        inputs=output,
        he_init=False
    )

    return output

def Decoder(name, latent_dim, hidden_dim, output_dim, upsample, latents):
    latents = T.clip(latents, lib.floatX(-50), lib.floatX(50))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Input',
        input_dim=latent_dim,
        output_dim=hidden_dim,
        filter_size=1,
        inputs=latents,
    ))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Conv1',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=CONV_FILTER_SIZE,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Conv2',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=CONV_FILTER_SIZE,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Conv3',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=CONV_FILTER_SIZE,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D(
        name+'.Conv4',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=CONV_FILTER_SIZE,
        inputs=output,
    ))

    if upsample:
        output = T.nnet.relu(lib.ops.deconv2d.Deconv2D(
            name+'.Up',
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            filter_size=CONV_FILTER_SIZE,
            inputs=output,
        ))

    output = lib.ops.conv2d.Conv2D(
        name+'.Output',
        input_dim=hidden_dim,
        output_dim=output_dim,
        filter_size=1,
        inputs=output,
        he_init=False
    )

    return output

def split(mu_and_log_sigma):
    return mu_and_log_sigma[:,::2], mu_and_log_sigma[:,1::2]

total_iters = T.iscalar('total_iters')
images = T.itensor4('images')

alpha = T.minimum(1, T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS))
beta = -10 # Prevent log_sigma from getting too tiny

# Layer 1

def E1(inputs):
    return Encoder('E1', N_CHANNELS*CONV_DIM, CONV_DIM, LATENT_DIM, False, inputs)

def D1(latents):
    return Decoder('D1', LATENT_DIM, CONV_DIM, 256*N_CHANNELS, False, latents)

embedded = lib.ops.embedding.Embedding(
    'Embedding', 
    256, 
    CONV_DIM, 
    images
)
embedded = embedded.dimshuffle(0,1,4,2,3)
embedded = embedded.reshape((
    embedded.shape[0], 
    embedded.shape[1] * embedded.shape[2], 
    embedded.shape[3], 
    embedded.shape[4]
))

mu_and_log_sigma1 = E1(embedded)
mu1, log_sigma1 = split(mu_and_log_sigma1)

if VANILLA:
    latents1 = mu1
else:
    eps = T.cast(theano_srng.normal(mu1.shape), theano.config.floatX)
    latents1 = mu1 + (eps * T.exp(log_sigma1))

outputs1 = D1(latents1)

reconst_cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(outputs1
        .reshape((-1,256,N_CHANNELS,HEIGHT,WIDTH))
        .dimshuffle(0,2,3,4,1)
        .reshape((-1, 256))
    ),
    images.flatten()
).mean()

# Layer 2

def E2(inputs):
    return Encoder('E2', 2*LATENT_DIM, L2_CONV_DIM, LATENT_DIM, True, inputs)

def D2(latents):
    return Decoder('D2', LATENT_DIM, L2_CONV_DIM, 2*LATENT_DIM, True, latents)

mu_and_log_sigma2 = E2(mu_and_log_sigma1)
mu2, log_sigma2 = split(mu_and_log_sigma2)

if VANILLA:
    latents2 = mu2
else:
    eps = T.cast(theano_srng.normal(mu2.shape), theano.config.floatX)
    latents2 = mu2 + (eps * T.exp(log_sigma2))

outputs2 = D2(latents2)
mu1_prior, log_sigma1_prior = split(outputs2)
log_sigma1_prior = T.maximum(beta, log_sigma1_prior)

kl_cost_1 = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
    mu1,
    log_sigma1,
    mu1_prior,
    log_sigma1_prior
).mean(axis=0).sum()

# Layer 3

def E3(inputs):
    inputs = lasagne.theano_extensions.padding.pad(
        inputs,
        width=1,
        batch_ndim=2
    )
    return Encoder('E3', 2*LATENT_DIM, L3_CONV_DIM, LATENT_DIM, True, inputs)

def D3(latents):
    result = Decoder('D3', LATENT_DIM, L3_CONV_DIM, 2*LATENT_DIM, True, latents)
    return result[:,:,1:-1,1:-1]

mu_and_log_sigma3 = E3(mu_and_log_sigma2)
mu3, log_sigma3 = split(mu_and_log_sigma3)

if VANILLA:
    latents3 = mu3
else:
    eps = T.cast(theano_srng.normal(mu3.shape), theano.config.floatX)
    latents3 = mu3 + (eps * T.exp(log_sigma3))

outputs3 = D3(latents3)
mu2_prior, log_sigma2_prior = split(outputs3)
log_sigma2_prior = T.maximum(beta, log_sigma2_prior)

kl_cost_2 = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
    mu2,
    log_sigma2,
    mu2_prior,
    log_sigma2_prior
).mean(axis=0).sum()

# Layer 4

def E4(inputs):
    return Encoder('E4', 2*LATENT_DIM, L4_CONV_DIM, LATENT_DIM, True, inputs)

def D4(latents):
    return Decoder('D4', LATENT_DIM, L4_CONV_DIM, 2*LATENT_DIM, True, latents)

mu_and_log_sigma4 = E4(mu_and_log_sigma3)
mu4, log_sigma4 = split(mu_and_log_sigma4)

if VANILLA:
    latents4 = mu4
else:
    eps = T.cast(theano_srng.normal(mu4.shape), theano.config.floatX)
    latents4 = mu4 + (eps * T.exp(log_sigma4))
    # latents4 = lib.debug.print_stats('latents4', latents4)

outputs4 = D4(latents4)
mu3_prior, log_sigma3_prior = split(outputs4)
log_sigma3_prior = T.maximum(beta, log_sigma3_prior)

kl_cost_3 = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
    mu3,
    log_sigma3,
    mu3_prior,
    log_sigma3_prior
).mean(axis=0).sum()

# Layer 5

def E5(inputs):
    inputs = inputs.reshape((inputs.shape[0], 4*4*2*LATENT_DIM))
    return lib.ops.mlp.MLP(
        'E5', 
        4*4*2*LATENT_DIM, 
        L5_FC_DIM,
        2*LATENT_DIM,
        5,
        inputs
    )

def D5(latents):
    latents = T.clip(latents, lib.floatX(-50), lib.floatX(50))
    output = lib.ops.mlp.MLP(
        'D5',
        LATENT_DIM,
        L5_FC_DIM,
        4*4*2*LATENT_DIM,
        5,
        latents
    )
    return output.reshape((-1, 2*LATENT_DIM, 4, 4))

mu_and_log_sigma5 = E5(mu_and_log_sigma4)
mu5, log_sigma5 = split(mu_and_log_sigma5)

if VANILLA:
    latents5 = mu5
else:
    eps = T.cast(theano_srng.normal(mu5.shape), theano.config.floatX)
    latents5 = mu5 + (eps * T.exp(log_sigma5))
    # latents4 = lib.debug.print_stats('latents4', latents4)

outputs5 = D5(latents5)
mu4_prior, log_sigma4_prior = split(outputs5)
log_sigma4_prior = T.maximum(beta, log_sigma4_prior)

kl_cost_4 = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
    mu4,
    log_sigma4,
    mu4_prior,
    log_sigma4_prior
).mean(axis=0).sum()

reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(mu5, log_sigma5).mean(axis=0).sum()

# Assembling everything together

kl_cost_1 /= float(N_CHANNELS * WIDTH * HEIGHT)
kl_cost_2 /= float(N_CHANNELS * WIDTH * HEIGHT)
kl_cost_3 /= float(N_CHANNELS * WIDTH * HEIGHT)
kl_cost_4 /= float(N_CHANNELS * WIDTH * HEIGHT)
reg_cost  /= float(N_CHANNELS * WIDTH * HEIGHT)

if VANILLA:
    cost = reconst_cost
else:
    cost = reconst_cost + (alpha * kl_cost_1) + ((alpha**2) * kl_cost_2)  + ((alpha**3) * kl_cost_3) + ((alpha**4) * kl_cost_4) + ((alpha**5) * reg_cost)

# Sampling

def randn(shape):
    return T.as_tensor_variable(
        np.random.normal(size=shape).astype(theano.config.floatX)
    )

z5_sample = randn((100, LATENT_DIM))
mu4_prior_sample, log_sigma4_prior_sample = split(D5(z5_sample))
z4_sample = T.cast(
    mu4_prior_sample + (T.exp(log_sigma4_prior_sample) * randn((100, LATENT_DIM, 4, 4))),
    theano.config.floatX
)
mu3_prior_sample, log_sigma3_prior_sample = split(D4(z4_sample))
z3_sample = T.cast(
    mu3_prior_sample + (T.exp(log_sigma3_prior_sample) * randn((100, LATENT_DIM, 8, 8))),
    theano.config.floatX
)
mu2_prior_sample, log_sigma2_prior_sample = split(D3(z3_sample))
z2_sample = T.cast(
    mu2_prior_sample + (T.exp(log_sigma2_prior_sample) * randn((100, LATENT_DIM, HEIGHT/2, WIDTH/2))),
    theano.config.floatX
)
mu1_prior_sample, log_sigma1_prior_sample = split(D2(z2_sample))
z1_sample = T.cast(
    mu1_prior_sample + (T.exp(log_sigma1_prior_sample) * randn((100, LATENT_DIM, HEIGHT, WIDTH))),
    theano.config.floatX
)
output_sample = lib.ops.softmax_and_sample.softmax_and_sample(
    D1(z1_sample)
        .reshape((-1,256,N_CHANNELS,HEIGHT,WIDTH))
        .dimshuffle(0,2,3,4,1)
)
if MODE=='256ary':
    sample_fn = theano.function(
        [],
        output_sample
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
        image.save('{}_{}.png'.format(filename, tag))

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
        ('kl1', kl_cost_1),
        ('kl2', kl_cost_2),
        ('kl3', kl_cost_3),
        ('kl4', kl_cost_4),
        ('reg', reg_cost)
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    # test_data=dev_data,
    callback=generate_and_save_samples,
    times=TIMES
)