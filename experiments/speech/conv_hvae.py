"""
Convolutional Hierarchical Variational Autoencoder
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
import lib.debug
import lib.train_loop
import lib.audio
import lib.ops.conv1d
import lib.ops.deconv1d
import lib.ops.mlp
import lib.ops.softmax_and_sample
import lib.ops.embedding
import lib.ops.kl_unit_gaussian
import lib.ops.kl_gaussian_gaussian
import lib.ops.grad_scale

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
import scipy.misc

import functools

Q_LEVELS = 256
GRAD_CLIP = 1
ALPHA_ITERS = 20000
BETA_ITERS = 100
VANILLA = False
LR = 2e-4

# Dataset
DATA_PATH = '/media/seagate/blizzard/parts'
N_FILES = 141703
# DATA_PATH = '/PersimmonData/kiwi_parts'
# N_FILES = 516
BITRATE = 16000

# Other constants
TIMES = ('iters', 100, 2000, 1000)
Q_ZERO = np.int32(Q_LEVELS//2)

# Hyperparams
BATCH_SIZE = 1
SEQ_LEN = 2**16
# BATCH_SIZE = 512
# SEQ_LEN = 128
EMBED_DIM = 32

L1_DIM = 32
L1_LATENT = 32

L2_DIM = 64
L2_LATENT = 64

# L3_DIM = 128
# L3_LATENT = 128

# L4_DIM = 512
# L4_LATENT = 512

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def EncoderB(name, input_dim, hidden_dim, latent_dim, inputs):
    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Input',
        input_dim=input_dim,
        output_dim=hidden_dim,
        filter_size=1,
        inputs=inputs,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv1S',
        input_dim=hidden_dim,
        output_dim=8*hidden_dim,
        filter_size=17,
        inputs=output,
        stride=8,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv2',
        input_dim=8*hidden_dim,
        output_dim=8*hidden_dim,
        filter_size=1,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv2',
        input_dim=8*hidden_dim,
        output_dim=8*hidden_dim,
        filter_size=1,
        inputs=output,
    ))

    output = lib.ops.conv1d.Conv1D(
        name+'.Output',
        input_dim=8*hidden_dim,
        output_dim=2*latent_dim,
        filter_size=1,
        inputs=output,
        he_init=False
    )

    return output

def DecoderB(name, latent_dim, hidden_dim, output_dim, latents):
    latents = T.clip(latents, lib.floatX(-50), lib.floatX(50))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Input',
        input_dim=latent_dim,
        output_dim=8*hidden_dim,
        filter_size=1,
        inputs=latents,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv1',
        input_dim=8*hidden_dim,
        output_dim=8*hidden_dim,
        filter_size=1,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv2',
        input_dim=8*hidden_dim,
        output_dim=8*hidden_dim,
        filter_size=1,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.deconv1d.Deconv1D(
    # output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv2U',
        input_dim=8*hidden_dim,
        output_dim=hidden_dim,
        filter_size=17,
        inputs=output,
        stride=8
    ))

    # output = T.nnet.relu(lib.ops.deconv1d.Deconv1D(
    # # output = T.nnet.relu(lib.ops.conv1d.Conv1D(
    #     name+'.Conv3U',
    #     input_dim=2*hidden_dim,
    #     output_dim=hidden_dim,
    #     filter_size=5,
    #     inputs=output,
    # ))

    output = lib.ops.conv1d.Conv1D(
        name+'.Output',
        input_dim=hidden_dim,
        output_dim=output_dim,
        filter_size=1,
        inputs=output,
        he_init=False
    )

    return output

def EncoderA(name, input_dim, hidden_dim, latent_dim, inputs):
    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Input',
        input_dim=input_dim,
        output_dim=hidden_dim,
        filter_size=1,
        inputs=inputs,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv1S',
        input_dim=hidden_dim,
        output_dim=2*hidden_dim,
        filter_size=5,
        inputs=output,
        stride=2,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv2S',
        input_dim=2*hidden_dim,
        output_dim=4*hidden_dim,
        filter_size=5,
        inputs=output,
        stride=2
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv3',
        input_dim=4*hidden_dim,
        output_dim=4*hidden_dim,
        filter_size=5,
        inputs=output,
    ))

    output = lib.ops.conv1d.Conv1D(
        name+'.Output',
        input_dim=4*hidden_dim,
        output_dim=2*latent_dim,
        filter_size=1,
        inputs=output,
        he_init=False
    )

    return output

def DecoderA(name, latent_dim, hidden_dim, output_dim, latents):
    latents = T.clip(latents, lib.floatX(-50), lib.floatX(50))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Input',
        input_dim=latent_dim,
        output_dim=4*hidden_dim,
        filter_size=1,
        inputs=latents,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv1',
        input_dim=4*hidden_dim,
        output_dim=4*hidden_dim,
        filter_size=5,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.deconv1d.Deconv1D(
    # output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv2U',
        input_dim=4*hidden_dim,
        output_dim=2*hidden_dim,
        filter_size=5,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.deconv1d.Deconv1D(
    # output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv3U',
        input_dim=2*hidden_dim,
        output_dim=hidden_dim,
        filter_size=5,
        inputs=output,
    ))

    output = lib.ops.conv1d.Conv1D(
        name+'.Output',
        input_dim=hidden_dim,
        output_dim=output_dim,
        filter_size=1,
        inputs=output,
        he_init=False
    )

    return output

Encoder = EncoderB
Decoder = DecoderB

def split(mu_and_logsig):
    return mu_and_logsig[:,::2], mu_and_logsig[:,1::2]

total_iters = T.iscalar('total_iters')
sequences = T.imatrix('sequences')
reset = T.iscalar('reset')

alpha = T.minimum(1, T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS))
alpha = alpha**2

def clamp_logsig(logsig):
    # return logsig
    beta = T.minimum(1, T.cast(total_iters, theano.config.floatX) / lib.floatX(BETA_ITERS))
    result = T.nnet.relu(logsig, alpha=beta)
    result = T.maximum(-5, result)
    return result

def scale_grads(x):
    return lib.ops.grad_scale.grad_scale(x, alpha)

# Layer 1

def E1(inputs):
    return Encoder('E1', EMBED_DIM, L1_DIM, L1_LATENT, inputs)

def D1(latents):
    return Decoder('D1', L1_LATENT, L1_DIM, 256, latents)

embedded = lib.ops.embedding.Embedding(
    'Embedding',
    256,
    EMBED_DIM,
    sequences
)
embedded = embedded.dimshuffle(0,2,1) # (batch, seq, dim) to (batch, dim, seq)

mu_and_logsig1 = E1(embedded)
mu1, logsig1 = split(mu_and_logsig1)

if VANILLA:
    latents1 = mu1
else:
    eps = T.cast(theano_srng.normal(mu1.shape), theano.config.floatX)
    latents1 = mu1 + (eps * T.exp(logsig1))

outputs1 = D1(latents1)

reconst_cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(outputs1
        .dimshuffle(0,2,1)
        .reshape((-1, 256))
    ),
    sequences.flatten()
).mean()

# Layer 2

def E2(inputs):
    return Encoder('E2', 2*L1_LATENT, L2_DIM, L2_LATENT, inputs)

def D2(latents):
    return Decoder('D2', L2_LATENT, L2_DIM, 2*L1_LATENT, latents)

gs_mu_and_logsig1 = scale_grads(mu_and_logsig1)
gs_mu1, gs_logsig1 = split(gs_mu_and_logsig1)

mu_and_logsig2 = E2(gs_mu_and_logsig1)
mu2, logsig2 = split(mu_and_logsig2)

if VANILLA:
    latents2 = mu2
else:
    eps = T.cast(theano_srng.normal(mu2.shape), theano.config.floatX)
    latents2 = mu2 + (eps * T.exp(logsig2))

outputs2 = D2(latents2)
mu1_prior, logsig1_prior = split(outputs2)
logsig1_prior = clamp_logsig(logsig1_prior)

kl_1_2 = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
    gs_mu1, 
    gs_logsig1, 
    mu1_prior, 
    logsig1_prior
).mean(axis=0).sum()

# Layer 3

# def E3(inputs):
#     return Encoder('E3', 2*L2_LATENT, L3_DIM, L3_LATENT, inputs)

# def D3(latents):
#     return Decoder('D3', L3_LATENT, L3_DIM, 2*L2_LATENT, latents)

# gs_mu_and_logsig2 = scale_grads(mu_and_logsig2)
# gs_mu2, gs_logsig2 = split(gs_mu_and_logsig2)

# mu_and_logsig3 = E3(gs_mu_and_logsig2)
# mu3, logsig3 = split(mu_and_logsig3)

# if VANILLA:
#     latents3 = mu3
# else:
#     eps = T.cast(theano_srng.normal(mu3.shape), theano.config.floatX)
#     latents3 = mu3 + (eps * T.exp(logsig3))

# outputs3 = D3(latents3)
# mu2_prior, logsig2_prior = split(outputs3)
# logsig2_prior = clamp_logsig(logsig2_prior)

# kl_2_3 = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
#     gs_mu2, 
#     gs_logsig2, 
#     mu2_prior, 
#     logsig2_prior
# ).mean(axis=0).sum()

# # Layer 4

# def E4(inputs):
#     return Encoder('E4', 2*L3_LATENT, L4_DIM, L4_LATENT, inputs)

# def D4(latents):
#     return Decoder('D4', L4_LATENT, L4_DIM, 2*L3_LATENT, latents)

# mu_and_logsig4 = E4(mu_and_logsig3)
# mu4, logsig4 = split(mu_and_logsig4)

# if VANILLA:
#     latents4 = mu4
# else:
#     eps = T.cast(theano_srng.normal(mu4.shape), theano.config.floatX)
#     latents4 = mu4 + (eps * T.exp(logsig4))

# outputs4 = D4(latents4)
# mu3_prior, logsig3_prior = split(outputs4)
# logsig3_prior = clamp_logsig(logsig3_prior)
# kl_3_4 = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
#     mu3, 
#     logsig3, 
#     mu3_prior, 
#     logsig3_prior
# ).mean(axis=0).sum()

# # Assembling everything together

gs_mu2, gs_logsig2 = split(scale_grads(mu_and_logsig2))
reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(
    gs_mu2, 
    gs_logsig2
).mean(axis=0).sum()

kl_1_2 /= float(SEQ_LEN)
# kl_2_3 /= float(SEQ_LEN)
# kl_3_4 /= float(SEQ_LEN)
reg_cost  /= float(SEQ_LEN)

if VANILLA:
    cost = reconst_cost
else:
    cost = (
        reconst_cost
        # + kl_1_2
        # + kl_2_3
        # + reg_cost
        # + (alpha * kl_1_2)
        # + (alpha**2 * kl_2_3)
        # + (alpha**3 * kl_3_4)
        # + (alpha**4 * reg_cost)
    )

# Sampling

def randn(shape, mu=None, log_sigma=None):
    result = T.as_tensor_variable(
        np.random.normal(size=shape).astype(theano.config.floatX)
    )
    if log_sigma:
        result *= T.exp(log_sigma)
    if mu:
        result += mu
    return result

z2_sample = randn((10, L2_LATENT, SEQ_LEN/64))

# mu3_prior_sample, logsig3_prior_sample = split(D4(z4_sample))
# z3_sample = randn((10, L3_LATENT, SEQ_LEN/8), mu3_prior_sample, logsig3_prior_sample)

# mu2_prior_sample, logsig2_prior_sample = split(D3(z3_sample))
# z2_sample = randn((10, L2_LATENT, SEQ_LEN/4), mu2_prior_sample, logsig2_prior_sample)

mu1_prior_sample, logsig1_prior_sample = split(D2(z2_sample))
z1_sample = randn((10, L1_LATENT, SEQ_LEN/8), mu1_prior_sample, logsig1_prior_sample)

output_sample = lib.ops.softmax_and_sample.softmax_and_sample(
    D1(z1_sample).dimshuffle(0,2,1)
)

sample_fn = theano.function(
    [],
    output_sample
)

def generate_and_save_samples(tag):
    def write_audio_file(name, data):
        data = data.astype('float32')
        data /= 255.
        data -= 0.5
        data *= 0.95
        scipy.io.wavfile.write(name+'.wav', BITRATE, data)

    samples = sample_fn()
    for i, sample in enumerate(samples):
        write_audio_file('sample_{}_{}'.format(tag, i), sample)

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

lib.train_loop.train_loop(
    inputs=[total_iters, sequences, reset],
    inject_total_iters=True,
    cost=cost,
    prints=[
        ('alpha', alpha),
        ('cost', cost),
        ('reconst', reconst_cost), 
        ('kl_1_2', kl_1_2),
        # ('kl_2_3', kl_2_3),
        # ('kl_3_4', kl_3_4),
        ('reg', reg_cost)
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    callback=generate_and_save_samples,
    times=TIMES
)