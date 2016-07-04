"""
Hierarchical Variational Autoencoder
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
import lib.audio
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
TIMES = ('iters', 1000, 1000*1000)
# TIMES = ('seconds', 60*60, 60*60*24)
# GENERATE_SAMPLES_AND_SAVE_PARAMS = True
Q_ZERO = np.int32(Q_LEVELS//2)

# Hyperparams
BATCH_SIZE = 64
FRAME_SIZE = 16
SEQ_LEN = FRAME_SIZE**3
# EMBED_DIM = 32
# L1_DIM = 128
# L1_LATENT = 16
# L2_DIM = 256
# L2_LATENT = 32

EMBED_DIM = 64

L1_DIM = 512
L1_LATENT = 64
L1_N_LAYERS = 6

L2_DIM = 1024
L2_LATENT = 128
L2_N_LAYERS = 6

L3_DIM = 2048
L3_LATENT = 256
L3_N_LAYERS = 6

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

total_iters = T.iscalar('total_iters')
sequences = T.imatrix('sequences')
reset = T.iscalar('reset')

alpha = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS)
)

alpha2 = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(3*ALPHA_ITERS)
)

alpha3 = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(9*ALPHA_ITERS)
)

sequences = sequences.reshape((-1, 4))

embedded = lib.ops.embedding.Embedding(
    'Embedding',
    n_symbols=Q_LEVELS,
    output_dim=EMBED_DIM,
    inputs=sequences
)

l1_inputs = embedded.reshape((-1, FRAME_SIZE*EMBED_DIM))

l1_mu_and_log_sigma = lib.ops.mlp.MLP(
    'L1Encoder',
    input_dim=FRAME_SIZE*EMBED_DIM,
    hidden_dim=L1_DIM,
    output_dim=2*L1_LATENT,
    n_layers=L1_N_LAYERS,
    inputs=embedded.reshape((-1, FRAME_SIZE*EMBED_DIM))
)
l1_mu, l1_log_sigma = l1_mu_and_log_sigma[:,::2], l1_mu_and_log_sigma[:,1::2]

if VANILLA:
    l1_latents = l1_mu
else:
    eps = T.cast(theano_srng.normal(l1_mu.shape), theano.config.floatX)
    l1_latents = l1_mu + (eps * T.exp(l1_log_sigma))

def L1Decoder(latents):
    outputs = lib.ops.mlp.MLP(
        'L1Decoder',
        input_dim=L1_LATENT,
        hidden_dim=L1_DIM,
        output_dim=FRAME_SIZE*EMBED_DIM,
        n_layers=L1_N_LAYERS,
        inputs=latents
    )
    outputs = outputs.reshape((-1, FRAME_SIZE, EMBED_DIM))
    outputs = lib.ops.linear.Linear('L1DecoderOutput', 
            input_dim=EMBED_DIM, 
            output_dim=Q_LEVELS, 
            inputs=outputs,
    )
    return outputs

outputs = L1Decoder(l1_latents)

reconst_cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(outputs.reshape((-1,Q_LEVELS))),
    sequences.flatten()
).mean()

reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(l1_mu, l1_log_sigma).sum()

# TODO maybe try disconnecting the grad of l1_mu_and_log_sigma here?
# l1_mu_and_log_sigma = theano.gradient.disconnected_grad(l1_mu_and_log_sigma)

l2_mu_and_log_sigma = lib.ops.mlp.MLP(
    'L2Encoder',
    input_dim=FRAME_SIZE*2*L1_LATENT,
    hidden_dim=L2_DIM,
    output_dim=2*L2_LATENT,
    n_layers=L2_N_LAYERS,
    inputs=l1_mu_and_log_sigma.reshape((-1, FRAME_SIZE*2*L1_LATENT))
)
l2_mu, l2_log_sigma = l2_mu_and_log_sigma[:,::2], l2_mu_and_log_sigma[:,1::2]

if VANILLA:
    l2_latents = l2_mu
else:
    eps = T.cast(theano_srng.normal(l2_mu.shape), theano.config.floatX)
    l2_latents = l2_mu + (eps * T.exp(l2_log_sigma))

def L2Decoder(latents):
    l2_outputs = lib.ops.mlp.MLP(
        'L2Decoder',
        input_dim=L2_LATENT,
        hidden_dim=L2_DIM,
        output_dim=FRAME_SIZE*2*L1_LATENT,
        n_layers=L2_N_LAYERS,
        inputs=latents
    )
    l2_outputs = l2_outputs.reshape((-1, 2*L1_LATENT))
    l2_out_mu, l2_out_log_sigma = l2_outputs[:,::2], l2_outputs[:,1::2]
    return (l2_out_mu, l2_out_log_sigma)

l2_out_mu, l2_out_log_sigma = L2Decoder(l2_latents)
# optimization hack to prevent this KL term from being massive in the beginning
l2_out_log_sigma = T.nnet.relu(l2_out_log_sigma, alpha=alpha2)

l2_reconst_cost = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
    l1_mu, l1_log_sigma,
    l2_out_mu, l2_out_log_sigma
).sum()

l2_reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(l2_mu, l2_log_sigma).sum()

############## Level 3

l3_mu_and_log_sigma = lib.ops.mlp.MLP(
    'L3Encoder',
    input_dim=FRAME_SIZE*2*L2_LATENT,
    hidden_dim=L3_DIM,
    output_dim=2*L3_LATENT,
    n_layers=L3_N_LAYERS,
    inputs=l2_mu_and_log_sigma.reshape((-1, FRAME_SIZE*2*L2_LATENT))
)
l3_mu, l3_log_sigma = l3_mu_and_log_sigma[:,::2], l3_mu_and_log_sigma[:,1::2]

if VANILLA:
    l3_latents = l3_mu
else:
    eps = T.cast(theano_srng.normal(l3_mu.shape), theano.config.floatX)
    l3_latents = l3_mu + (eps * T.exp(l3_log_sigma))

def L3Decoder(latents):
    l3_outputs = lib.ops.mlp.MLP(
        'L3Decoder',
        input_dim=L3_LATENT,
        hidden_dim=L3_DIM,
        output_dim=FRAME_SIZE*2*L2_LATENT,
        n_layers=L3_N_LAYERS,
        inputs=latents
    )
    l3_outputs = l3_outputs.reshape((-1, 2*L2_LATENT))
    l3_out_mu, l3_out_log_sigma = l3_outputs[:,::2], l3_outputs[:,1::2]
    return (l3_out_mu, l3_out_log_sigma)

l3_out_mu, l3_out_log_sigma = L3Decoder(l3_latents)
# optimization hack to prevent this KL term from being massive in the beginning
l3_out_log_sigma = T.nnet.relu(l3_out_log_sigma, alpha=alpha3)

l3_reconst_cost = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
    l2_mu, l2_log_sigma,
    l3_out_mu, l3_out_log_sigma
).sum()

l3_reg_cost = lib.ops.kl_unit_gaussian.kl_unit_gaussian(l3_mu, l3_log_sigma).sum()


############## End level 3


reg_cost /= T.cast(sequences.flatten().shape[0], theano.config.floatX)
l2_reconst_cost /= T.cast(sequences.flatten().shape[0], theano.config.floatX)
l2_reg_cost /= T.cast(sequences.flatten().shape[0], theano.config.floatX)
l3_reconst_cost /= T.cast(sequences.flatten().shape[0], theano.config.floatX)
l3_reg_cost /= T.cast(sequences.flatten().shape[0], theano.config.floatX)

if VANILLA:
    cost = reconst_cost
else:
    cost = ((alpha3**3) * l3_reg_cost) + ((alpha2**2) * l3_reconst_cost) + (alpha * l2_reconst_cost) + reconst_cost

### Sampling code

N_SAMPLES = 100

z3 = theano_srng.normal((N_SAMPLES, L3_LATENT))
sample_l3_out_mu, sample_l3_out_log_sigma = L3Decoder(z3)

z2_eps = T.cast(theano_srng.normal(sample_l3_out_mu.shape), theano.config.floatX)
z2 = sample_l3_out_mu + (z2_eps * T.exp(sample_l3_out_log_sigma))
z2 = z2.reshape((-1, L2_LATENT))
sample_l2_out_mu, sample_l2_out_log_sigma = L2Decoder(z2)

z1_eps = T.cast(theano_srng.normal(sample_l2_out_mu.shape), theano.config.floatX)
z1 = sample_l2_out_mu + (z1_eps * T.exp(sample_l2_out_log_sigma))
z1 = z1.reshape((-1, L1_LATENT))
outs = L1Decoder(z1).reshape((N_SAMPLES, FRAME_SIZE**3, Q_LEVELS))

sample_fn = theano.function(
    [],
    lib.ops.softmax_and_sample.softmax_and_sample(outs)
)

def generate_and_save_samples(tag):
    def write_audio_file(name, data):
        data = data.astype('float32')
        data /= 255.
        data -= 0.5
        data *= 0.95
        scipy.io.wavfile.write(name+'.wav', BITRATE, data)

    samples = sample_fn().reshape((-1,))
    write_audio_file('sample_{}'.format(tag), samples)

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
        ('nll', l3_reg_cost + l3_reconst_cost + l2_reconst_cost + reconst_cost),
        ('l3reg', l3_reg_cost),
        ('l3reconst', l3_reconst_cost),
        ('l2reconst', l2_reconst_cost),
        ('reconst', reconst_cost), 
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    callback=generate_and_save_samples,
    times=TIMES
)