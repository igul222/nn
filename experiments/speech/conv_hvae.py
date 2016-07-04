"""
Convolutional Hierarchical Variational Autoencoder
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
import lib.audio
import lib.ops.conv1d
import lib.ops.deconv1d
import lib.ops.mlp
import lib.ops.softmax_and_sample
import lib.ops.embedding
import lib.ops.kl_unit_gaussian
import lib.ops.kl_gaussian_gaussian

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
import scipy.misc

import functools

Q_LEVELS = 256
GRAD_CLIP = 1
ALPHA_ITERS = 10000
VANILLA = False
LR = 2e-4

# Dataset
DATA_PATH = '/media/seagate/blizzard/parts'
N_FILES = 141703
# DATA_PATH = '/PersimmonData/kiwi_parts'
# N_FILES = 516
BITRATE = 16000

# Other constants
TIMES = ('iters', 1000, 100000, 1000)
Q_ZERO = np.int32(Q_LEVELS//2)

# Hyperparams
BATCH_SIZE = 128
SEQ_LEN = 512
EMBED_DIM = 256
L1_DIM = EMBED_DIM
L1_LATENT = EMBED_DIM

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def Encoder(name, input_dim, hidden_dim, latent_dim, downsample, inputs):

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Input',
        input_dim=input_dim,
        output_dim=hidden_dim,
        filter_size=5,
        inputs=inputs,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv1',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=5,
        inputs=output
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv2',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=5,
        inputs=output
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv3',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=5,
        inputs=output
    ))

    output = lib.ops.conv1d.Conv1D(
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

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Input',
        input_dim=latent_dim,
        output_dim=hidden_dim,
        filter_size=1,
        inputs=latents,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv1',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=5,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv2',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=5,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv3',
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        filter_size=5,
        inputs=output,
    ))

    output = T.nnet.relu(lib.ops.conv1d.Conv1D(
        name+'.Conv4',
        input_dim=hidden_dim,
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

def split(mu_and_log_sigma):
    return mu_and_log_sigma[:,::2], mu_and_log_sigma[:,1::2]


total_iters = T.iscalar('total_iters')
sequences = T.imatrix('sequences')
reset = T.iscalar('reset')

alpha = T.minimum(1, T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS))

# Layer 1

def E1(inputs):
    return Encoder('E1', EMBED_DIM, L1_DIM, L1_LATENT, True, inputs)

def D1(latents):
    return Decoder('D1', L1_LATENT, L1_DIM, 256, True, latents)

embedded = lib.ops.embedding.Embedding(
    'Embedding',
    256,
    EMBED_DIM,
    sequences
)
embedded = embedded.dimshuffle(0,2,1) # (batch, seq) to (batch, dim, seq)

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
        .dimshuffle(0,2,1)
        .reshape((-1, 256))
    ),
    sequences.flatten()
).mean()

reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(mu1, log_sigma1).mean(axis=0).sum()

# Assembling everything together

reg_cost  /= float(SEQ_LEN)

if VANILLA:
    cost = reconst_cost
else:
    cost = reconst_cost + (alpha * reg_cost)

# Sampling

def randn(shape):
    return T.as_tensor_variable(
        np.random.normal(size=shape).astype(theano.config.floatX)
    )

z1_sample = randn((10, L1_LATENT, SEQ_LEN/16))
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
        ('reg', reg_cost),
        # ('l2reg', l2_reg_cost),
        # ('l2reconst', l2_reconst_cost),
        ('reconst', reconst_cost), 
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    callback=generate_and_save_samples,
    times=TIMES
)