"""
Variational two-level speech generative model (never worked that great)
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
import lib.ops.conv_encoder
import lib.ops.conv_decoder
import lib.ops.kl_gaussian_gaussian
import lib.ops.conv2d
import lib.ops.diagonal_bilstm
import lib.ops.relu
import lib.ops.softmax_and_sample
import lib.ops.embedding
import lib.ops.gru

import numpy as np
import theano
# theano.config.profile = True
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
import lasagne

import functools

# Hyperparams
BATCH_SIZE = 128
FRAME_SIZE = 64
N_FRAMES = 16
SEQ_LEN = FRAME_SIZE*N_FRAMES

CONV_DIM = 16
CONV_N_POOLS = 3
CONV_FILTER_SIZE = 5
BIG_FC_DIM = 1024
SMALL_FC_DIM = 128
MLP_LAYERS = 4
LATENT_DIM = 256

Q_LEVELS = 256
GRAD_CLIP = 1
ALPHA_ITERS = 20000
VANILLA = False
LR = 2e-4

# Dataset
DATA_PATH = '/media/seagate/blizzard/parts'
N_FILES = 141703
# DATA_PATH = '/PersimmonData/kiwi_parts'
# N_FILES = 516
BITRATE = 16000

# Other constants
TIMES = ('iters', 5000, 100*1000)
# TIMES = ('seconds', 60*60, 60*60*24)
GENERATE_SAMPLES_AND_SAVE_PARAMS = True
Q_ZERO = np.int32(Q_LEVELS//2)
SAMPLE_LEN = 5*BITRATE
# SAMPLE_LEN = 1024

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)

def FrameProcessor(inputs):
    """
    inputs.shape: (batch size, FRAME_SIZE)
    output.shape: (batch size, BIG_FC_DIM)
    """
    batch_size = inputs.shape[0]
    n_frames = inputs.shape[1]

    inputs = lib.ops.embedding.Embedding(
        'FrameProcessor.Embedding', 
        Q_LEVELS, 
        CONV_DIM, 
        inputs
    )

    # output = inputs.reshape((-1, CONV_DIM*FRAME_SIZE))
    # return lib.ops.mlp.MLP(
    #     'FrameProcessor.MLP',
    #     input_dim=CONV_DIM*FRAME_SIZE,
    #     hidden_dim=BIG_FC_DIM,
    #     output_dim=BIG_FC_DIM,
    #     n_layers=MLP_LAYERS,
    #     inputs=output
    # )

    inputs = inputs.dimshuffle(0,2,1) # to (batch_size, n_channels, width)
    return lib.ops.conv_encoder.ConvEncoder(
        'FrameProcessor.ConvEncoder',
        input_n_channels=CONV_DIM,
        input_size=FRAME_SIZE,
        n_pools=CONV_N_POOLS,
        base_n_filters=CONV_DIM,
        filter_size=CONV_FILTER_SIZE,
        output_dim=BIG_FC_DIM,
        inputs=inputs,
        mode='1d',
        deep=False
    )

def Prior(contexts):
    """
    contexts.shape: (batch size, BIG_FC_DIM)
    outputs: (mu, log_sigma), each with shape (batch size, LATENT_DIM)
    """
    mu_and_log_sigma = lib.ops.mlp.MLP(
        'Prior', 
        input_dim=BIG_FC_DIM, 
        hidden_dim=BIG_FC_DIM, 
        output_dim=2*LATENT_DIM, 
        n_layers=MLP_LAYERS, 
        inputs=contexts
    )
    return mu_and_log_sigma[:,::2], mu_and_log_sigma[:,1::2]

def Encoder(processed_frames, contexts):
    """
    processed_frames.shape: (batch size, BIG_FC_DIM)
    contexts.shape: (batch size, BIG_FC_DIM)
    outputs: (mu, log_sigma), each with shape (batch size, n frames, LATENT_DIM)
    """
    inputs = T.concatenate([
        processed_frames,
        contexts
    ], axis=1)

    mu_and_log_sigma = lib.ops.mlp.MLP(
        'Encoder', 
        input_dim=2*BIG_FC_DIM, 
        hidden_dim=BIG_FC_DIM, 
        output_dim=2*LATENT_DIM, 
        n_layers=MLP_LAYERS, 
        inputs=inputs,
    )
    return mu_and_log_sigma[:,::2], mu_and_log_sigma[:,1::2]

def FrameDecoder(latents, contexts):
    """
    latents.shape: (batch size, LATENT_DIM)
    contexts.shape: (batch size, BIG_FC_DIM)
    output: (batch size, FRAME_SIZE, SMALL_FC_DIM)
    """
    inputs = T.concatenate([
        latents,
        contexts
    ], axis=1)

    inputs = lib.ops.mlp.MLP(
        'FrameDecoder.MLP',
        input_dim=BIG_FC_DIM+LATENT_DIM,
        hidden_dim=BIG_FC_DIM,
        output_dim=BIG_FC_DIM,
        n_layers=MLP_LAYERS,
        inputs=inputs
    )

    return lib.ops.conv_decoder.ConvDecoder(
        'FrameDecoder.ConvDecoder',
        input_dim=BIG_FC_DIM,
        n_unpools=CONV_N_POOLS,
        base_n_filters=CONV_DIM,
        filter_size=CONV_FILTER_SIZE,
        output_size=FRAME_SIZE,
        output_n_channels=SMALL_FC_DIM,
        inputs=inputs,
        mode='1d',
        deep=False
    ).dimshuffle(0,2,1) # (batch, channels, width) to (batch, width, channels)

def SampleDecoder(decoded_latents, prev_samples, reset):
    """
    frame_decoder_outputs.shape: (batch size, seq len, SMALL_FC_DIM)
    prev_samples.shape: (batch size, seq len)
    reset: iscalar
    """
    embedded_prev_samples = lib.ops.embedding.Embedding(
        'SampleDecoder.Embedding',
        n_symbols=Q_LEVELS,
        output_dim=SMALL_FC_DIM,
        inputs=prev_samples
    )

    output = T.concatenate([
        decoded_latents, 
        embedded_prev_samples
    ], axis=2)

    output = lib.ops.gru.GRU(
        'SampleDecoder.GRU',
        input_dim=2*SMALL_FC_DIM,
        hidden_dim=SMALL_FC_DIM,
        inputs=output,
        reset=reset
    )

    output = lib.ops.linear.Linear(
        'SampleDecoder.Output',
        input_dim=SMALL_FC_DIM,
        output_dim=Q_LEVELS,
        inputs=output
    )

    return output

def FrameRecurrence(processed_frames, reset):
    """
    processed_frames.shape: (batch size, n frames, BIG_FC_DIM)
    reset: iscalar
    output.shape: (batch size, n frames, BIG_FC_DIM)
    """
    output = processed_frames

    output = lib.ops.gru.GRU(
        'Recurrence.GRU1',
        input_dim=BIG_FC_DIM,
        hidden_dim=BIG_FC_DIM,
        inputs=output,
        reset=reset
    )

    output = lib.ops.gru.GRU(
        'Recurrence.GRU2',
        input_dim=BIG_FC_DIM,
        hidden_dim=BIG_FC_DIM,
        inputs=output,
        reset=reset
    )

    output = lib.ops.gru.GRU(
        'Recurrence.GRU3',
        input_dim=BIG_FC_DIM,
        hidden_dim=BIG_FC_DIM,
        inputs=output,
        reset=reset
    )

    return output

total_iters = T.iscalar('total_iters')
sequences = T.imatrix('sequences')
reset = T.iscalar('reset')

batch_size = sequences.shape[0]
n_frames = sequences.shape[1] / FRAME_SIZE

frames = sequences.reshape((-1, n_frames, FRAME_SIZE))

processed_frames = FrameProcessor(
    frames.reshape((-1, FRAME_SIZE))
).reshape((batch_size, n_frames, -1))

contexts = FrameRecurrence(processed_frames[:,:-1], reset)

mu_prior, log_sigma_prior = Prior(contexts.reshape((-1, BIG_FC_DIM)))

mu_post, log_sigma_post = Encoder(
    processed_frames[:,1:].reshape((-1, BIG_FC_DIM)),
    contexts.reshape((-1,BIG_FC_DIM))
)

if VANILLA:
    latents = mu_post
else:
    eps = T.cast(theano_srng.normal(mu_post.shape), theano.config.floatX)
    latents = mu_post + (eps * T.exp(log_sigma_post))

decoded_latents = FrameDecoder(
    latents, 
    contexts.reshape((-1, BIG_FC_DIM))
)

outputs = SampleDecoder(
    decoded_latents.reshape((batch_size, -1, SMALL_FC_DIM)),
    sequences[:,FRAME_SIZE-1:-1],
    reset
)

reconst_cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(outputs.reshape((-1,Q_LEVELS))),
    frames[:,1:].flatten()
).mean()

reg_cost = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
    mu_post, 
    log_sigma_post, 
    mu_prior, 
    log_sigma_prior
).sum()
reg_cost /= T.cast(frames[:,1:].flatten().shape[0], theano.config.floatX)

alpha = T.minimum(
    1,
    T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS)
)

if VANILLA:
    cost = reconst_cost
else:
    cost = reconst_cost + (alpha * reg_cost)


sample_fn_contexts = FrameRecurrence(processed_frames, reset)
sample_fn_mu, sample_fn_log_sigma = Prior(
    sample_fn_contexts.reshape((-1, BIG_FC_DIM))
)
sample_fn_latents = sample_fn_mu + (
    T.cast(
        theano_srng.normal(sample_fn_mu.shape), 
        theano.config.floatX
    ) * T.exp(sample_fn_log_sigma)
)
sample_decoded_latents_fn = theano.function(
    [sequences, reset],
    FrameDecoder(
        sample_fn_latents,
        sample_fn_contexts.reshape((-1, BIG_FC_DIM))
    ),
    on_unused_input='warn'
)

sample_fn_prev_sample = T.ivector('sample_fn_prev_sample')
sample_fn_decoded_latents_timestep = T.matrix('sample_fn_decoded_latents_timestep')
sample_decoder_fn = theano.function(
    [sample_fn_decoded_latents_timestep, sample_fn_prev_sample, reset],
    lib.ops.softmax_and_sample.softmax_and_sample(SampleDecoder(
        sample_fn_decoded_latents_timestep[:,None,:],
        sample_fn_prev_sample[:,None],
        reset
    )),
    on_unused_input='warn'
)

def generate_and_save_samples(tag):
    if not GENERATE_SAMPLES_AND_SAVE_PARAMS:
        return

    def write_audio_file(name, data):
        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95
        scipy.io.wavfile.write(name+'.wav', BITRATE, data)

    # Generate 10 sample files, each 5 seconds long
    N_SEQS = 10
    LENGTH = SAMPLE_LEN - (SAMPLE_LEN%FRAME_SIZE)

    samples = np.zeros(
        (N_SEQS, LENGTH), 
        dtype='int32'
    )
    samples[:, :FRAME_SIZE] = Q_ZERO

    for i in xrange(FRAME_SIZE, LENGTH):
        if i % FRAME_SIZE == 0:
            decoded_latents = sample_decoded_latents_fn(
                samples[:,i-FRAME_SIZE:i],
                np.int32(i == FRAME_SIZE)
            )
        samples[:, i:i+1] = sample_decoder_fn(
            decoded_latents[:, i % FRAME_SIZE],
            samples[:, i-1],
            np.int32(i == FRAME_SIZE)
        )

    for i in xrange(N_SEQS):
        write_audio_file("sample_{}_{}".format(tag, i), samples[i].reshape((-1)))


train_data = functools.partial(
    lib.audio.feed_epoch,
    DATA_PATH,
    N_FILES,
    BATCH_SIZE,
    SEQ_LEN,
    FRAME_SIZE,
    Q_LEVELS,
    Q_ZERO
)

lib.train_loop.train_loop(
    inputs=[total_iters, sequences, reset],
    inject_total_iters=True,
    cost=cost,
    prints=[
        ('alpha', alpha), 
        ('reconst', reconst_cost), 
        ('reg', reg_cost)
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    callback=generate_and_save_samples,
    times=TIMES
)