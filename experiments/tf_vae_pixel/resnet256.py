"""
Multilayer VAE + Pixel CNN
Ishaan Gulrajani
"""

import os, sys
if 'ISHAAN_NN_LIB' in os.environ:
    sys.path.append(os.environ['ISHAAN_NN_LIB'])
else:
    sys.path.append(os.getcwd())

N_GPUS = 8

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=N_GPUS)
except ImportError:
    pass

import tflib as lib
import tflib.debug
import tflib.train_loop_2
import tflib.ops.kl_unit_gaussian
import tflib.ops.kl_gaussian_gaussian
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.ops.embedding

import tflib.lsun256
import tflib.lsun256_test

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

# lsun256, lsun256_test
DATASET = 'lsun256'

DIM_EMBED     = 16
DIM_256       = 64
DIM_256_PIX   = 64
DIM_128       = 64
DIM_64        = 128
DIM_64_LATENT = 32
DIM_64_PIX    = 256
DIM_32       = 256
DIM_16       = 512
DIM_8        = 512
DIM_8_LATENT = 64
DIM_8_PIX    = 512
DIM_4        = 512
DIM_1_LATENT = 512

TIMES = {
    'mode': 'iters',
    'print_every': 1,
    'test_every': 10000,
    'stop_after': 400000
}

LR = 2e-4

LR_DECAY_AFTER = 175000
LR_DECAY_FACTOR = .5

ALPHA1_ITERS = 5000
ALPHA2_ITERS = 10000
ALPHA3_ITERS = 20000
KL_PENALTY = 1.01
BETA_ITERS = 1000

BATCH_SIZE = 32
N_CHANNELS = 3
HEIGHT = 256
WIDTH = 256
LATENTS1_WIDTH = 64
LATENTS1_HEIGHT = 64
LATENTS2_WIDTH = 8
LATENTS2_HEIGHT = 8

if DATASET=='lsun256_test':
    train_data, dev_data = lib.lsun256_test.load(BATCH_SIZE)
elif DATASET=='lsun256':
    train_data = lib.lsun256.load(BATCH_SIZE)

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

def nonlinearity(x):
    return tf.nn.elu(x)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, inputs, inputs_stdev, filter_size, mask_type=None, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if mask_type != None and resample != None:
        raise Exception('Unsupported configuration')

    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, stride=2)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(SubpixelConv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, mask_type=mask_type, he_init=False, biases=True, inputs=inputs)

    output = inputs
    if mask_type == None:
        output = nonlinearity(output)
        output = conv_1(name+'.Conv1', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
        output = nonlinearity(output)
        output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
    else:
        output = nonlinearity(output)
        output_a = conv_1(name+'.Conv1A', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
        output_b = conv_1(name+'.Conv1B', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
        output = pixcnn_gated_nonlinearity(output_a, output_b)
        output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)

    return shortcut + (0.3 * output)

def Enc1(images):
    output = images
    output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_256, filter_size=1, inputs=output, he_init=False)
    output = ResidualBlock('Enc1.Res1', input_dim=DIM_256, output_dim=DIM_128, filter_size=3, resample='down', inputs_stdev=1, inputs=output)
    output = ResidualBlock('Enc1.Res2', input_dim=DIM_128, output_dim=DIM_128, filter_size=3, resample=None, inputs_stdev=1, inputs=output)
    output = ResidualBlock('Enc1.Res3', input_dim=DIM_128, output_dim=DIM_64, filter_size=3, resample='down', inputs_stdev=1, inputs=output)
    output = ResidualBlock('Enc1.Res4', input_dim=DIM_64, output_dim=DIM_64, filter_size=3, resample=None, inputs_stdev=1, inputs=output)
    output = lib.ops.conv2d.Conv2D('Enc1.Out', input_dim=DIM_64, output_dim=2*DIM_64_LATENT, filter_size=1, inputs=output, he_init=False)
    return output

def Dec1(latents, images):
    output = tf.clip_by_value(latents, -15., 15.)

    output = lib.ops.conv2d.Conv2D('Dec1.Input', input_dim=DIM_64_LATENT, output_dim=DIM_64, filter_size=1, inputs=output, he_init=False)
    output = ResidualBlock('Dec1.Res1', input_dim=DIM_64, output_dim=DIM_64, filter_size=3, resample=None, inputs_stdev=1, inputs=output)
    output = ResidualBlock('Dec1.Res2', input_dim=DIM_64, output_dim=DIM_128, filter_size=3, resample='up', inputs_stdev=1, inputs=output)
    output = ResidualBlock('Dec1.Res3', input_dim=DIM_128, output_dim=DIM_128, filter_size=3, resample=None, inputs_stdev=1, inputs=output)
    output = ResidualBlock('Dec1.Res4', input_dim=DIM_128, output_dim=DIM_256, filter_size=3, resample='up', inputs_stdev=1, inputs=output)

    masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_256_PIX, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)

    # Make the stdev of output and masked_images match
    output /= np.sqrt(4)
    # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
    output = tf.concat(1, [masked_images, output])

    output = ResidualBlock('Dec1.Pix2Res', input_dim=DIM_256_PIX+DIM_256, output_dim=DIM_256_PIX, filter_size=3, mask_type=('b', N_CHANNELS), inputs_stdev=1, inputs=output)
    output = ResidualBlock('Dec1.Pix3Res', input_dim=DIM_256_PIX, output_dim=DIM_256_PIX, filter_size=1, mask_type=('b', N_CHANNELS), inputs_stdev=1, inputs=output)

    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_256_PIX, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)

    return tf.transpose(
        tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
        [0,2,3,4,1]
    )

def Enc2(latents):
    output = tf.clip_by_value(latents, -15., 15.)

    output = lib.ops.conv2d.Conv2D('Enc2.Input', input_dim=DIM_64_LATENT, output_dim=DIM_64, filter_size=1, inputs=output, he_init=False)
    output = ResidualBlock('Enc2.InputRes0', input_dim=DIM_64, output_dim=DIM_64, filter_size=3, resample=None, inputs_stdev=1, inputs=output)
    output = ResidualBlock('Enc2.InputRes', input_dim=DIM_64, output_dim=DIM_32, filter_size=3, resample='down', inputs_stdev=1, inputs=output)

    output = ResidualBlock('Enc2.Res1Pre', input_dim=DIM_32, output_dim=DIM_32, filter_size=3, resample=None, inputs_stdev=1,          inputs=output)
    output = ResidualBlock('Enc2.Res1', input_dim=DIM_32, output_dim=DIM_16, filter_size=3, resample='down', inputs_stdev=1,          inputs=output)
    output = ResidualBlock('Enc2.Res2Pre', input_dim=DIM_16, output_dim=DIM_16, filter_size=3, resample=None, inputs_stdev=1,          inputs=output)
    output = ResidualBlock('Enc2.Res2', input_dim=DIM_16, output_dim=DIM_8, filter_size=3, resample='down', inputs_stdev=np.sqrt(2), inputs=output)
    output = ResidualBlock('Enc2.Res3Pre', input_dim=DIM_8, output_dim=DIM_8, filter_size=3, resample=None, inputs_stdev=1,          inputs=output)
    output = ResidualBlock('Enc2.Res3', input_dim=DIM_8, output_dim=DIM_8, filter_size=3, resample=None,   inputs_stdev=np.sqrt(3), inputs=output)

    output = lib.ops.conv2d.Conv2D('Enc2.Out', input_dim=DIM_8, output_dim=2*DIM_8_LATENT, filter_size=1, inputs=output, he_init=False)

    return output

def Dec2(latents, targets):
    output = tf.clip_by_value(latents, -15., 15.)

    output = lib.ops.conv2d.Conv2D('Dec2.Input', input_dim=DIM_8_LATENT, output_dim=DIM_8, filter_size=1, inputs=output, he_init=False)

    output = ResidualBlock('Dec2.Res1', input_dim=DIM_8, output_dim=DIM_8, filter_size=3, resample=None, inputs_stdev=1, inputs=output)
    output = ResidualBlock('Dec2.Res1Post', input_dim=DIM_8, output_dim=DIM_8, filter_size=3, resample=None, inputs_stdev=1, inputs=output)
    output = ResidualBlock('Dec2.Res2', input_dim=DIM_8, output_dim=DIM_16, filter_size=3, resample='up', inputs_stdev=np.sqrt(2), inputs=output)
    output = ResidualBlock('Dec2.Res2Post', input_dim=DIM_16, output_dim=DIM_16, filter_size=3, resample=None, inputs_stdev=np.sqrt(2), inputs=output)
    output = ResidualBlock('Dec2.Res3', input_dim=DIM_16, output_dim=DIM_32, filter_size=3, resample='up', inputs_stdev=np.sqrt(3), inputs=output)
    output = ResidualBlock('Dec2.Res3Post', input_dim=DIM_32, output_dim=DIM_32, filter_size=3, resample=None, inputs_stdev=np.sqrt(3), inputs=output)

    output = ResidualBlock('Dec2.Res4', input_dim=DIM_32, output_dim=DIM_64, filter_size=3, resample='up', inputs_stdev=np.sqrt(3), inputs=output)
    output = ResidualBlock('Dec2.Res4Post', input_dim=DIM_64, output_dim=DIM_64, filter_size=3, resample=None, inputs_stdev=np.sqrt(3), inputs=output)

    masked_targets = lib.ops.conv2d.Conv2D('Dec2.Pix1', input_dim=DIM_64_LATENT, output_dim=DIM_64, filter_size=5, inputs=targets, mask_type=('a', 1), he_init=False)

    # Make the stdev of output and masked_targets match
    output /= np.sqrt(4)

    # Warning! Because of the masked convolutions it's very important that masked_targets comes first in this concat
    output = tf.concat(1, [masked_targets, output])

    output = ResidualBlock('Dec2.Pix2Res', input_dim=2*DIM_64, output_dim=DIM_64_PIX, filter_size=3, mask_type=('b', 1), inputs_stdev=1, inputs=output)
    output = ResidualBlock('Dec2.Pix3Res', input_dim=DIM_64_PIX,   output_dim=DIM_64_PIX, filter_size=3, mask_type=('b', 1), inputs_stdev=1, inputs=output)
    output = ResidualBlock('Dec2.Pix4Res', input_dim=DIM_64_PIX,   output_dim=DIM_64_PIX, filter_size=3, mask_type=('b', 1), inputs_stdev=1, inputs=output)

    output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=DIM_64_PIX, output_dim=2*DIM_64_LATENT, filter_size=1, mask_type=('b', 1), he_init=False, inputs=output)

    return output

def Enc3(latents):
    output = tf.clip_by_value(latents, -15., 15.)

    output = lib.ops.conv2d.Conv2D('Enc3.Input', input_dim=DIM_8_LATENT, output_dim=DIM_8, filter_size=1, inputs=output, he_init=False)

    output = ResidualBlock('Enc3.Res0', input_dim=DIM_8, output_dim=DIM_8, filter_size=3, resample=None, inputs_stdev=1,          he_init=True, inputs=output)
    output = ResidualBlock('Enc3.Res1Pre', input_dim=DIM_8, output_dim=DIM_8, filter_size=3, resample=None, inputs_stdev=1,          he_init=True, inputs=output)
    output = ResidualBlock('Enc3.Res1', input_dim=DIM_8, output_dim=DIM_4, filter_size=3, resample='down', inputs_stdev=1,          he_init=True, inputs=output)
    output = ResidualBlock('Enc3.Res2Pre', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None,   inputs_stdev=np.sqrt(2), he_init=True, inputs=output)
    output = ResidualBlock('Enc3.Res2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None,   inputs_stdev=np.sqrt(2), he_init=True, inputs=output)

    output = tf.reshape(output, [-1, 4*4*DIM_4])
    output = lib.ops.linear.Linear('Enc3.Output', input_dim=4*4*DIM_4, output_dim=2*DIM_1_LATENT, inputs=output)

    return output

def Dec3(latents, targets):
    output = tf.clip_by_value(latents, -15., 15.)
    output = lib.ops.linear.Linear('Dec3.Input', input_dim=DIM_1_LATENT, output_dim=4*4*DIM_4, inputs=output)

    output = tf.reshape(output, [-1, DIM_4, 4, 4])

    output = ResidualBlock('Dec3.Res1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, inputs_stdev=np.sqrt(3), he_init=True, inputs=output)
    output = ResidualBlock('Dec3.Res1Post', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, inputs_stdev=np.sqrt(3), he_init=True, inputs=output)
    output = ResidualBlock('Dec3.Res3', input_dim=DIM_4, output_dim=DIM_8, filter_size=3, resample='up', inputs_stdev=np.sqrt(3), he_init=True, inputs=output)
    output = ResidualBlock('Dec3.Res3Post', input_dim=DIM_8, output_dim=DIM_8, filter_size=3, resample=None, inputs_stdev=np.sqrt(3), he_init=True, inputs=output)
    output = ResidualBlock('Dec3.Res3Post', input_dim=DIM_8, output_dim=DIM_8, filter_size=3, resample=None, inputs_stdev=np.sqrt(3), he_init=True, inputs=output)


    masked_targets = lib.ops.conv2d.Conv2D('Dec3.Pix1', input_dim=DIM_8_LATENT, output_dim=DIM_8, filter_size=5, mask_type=('a', 1), he_init=False, inputs=targets)

    # Make the stdev of output and masked_targets match
    output /= np.sqrt(4)

    output = tf.concat(1, [masked_targets, output])

    output = ResidualBlock('Dec3.Pix2Res', input_dim=2*DIM_8, output_dim=DIM_8_PIX, filter_size=3, mask_type=('b', 1), inputs_stdev=1, he_init=True, inputs=output)
    output = ResidualBlock('Dec3.Pix3Res', input_dim=DIM_8_PIX, output_dim=DIM_8_PIX, filter_size=3, mask_type=('b', 1), inputs_stdev=np.sqrt(2), he_init=True, inputs=output)
    output = ResidualBlock('Dec3.Pix4Res', input_dim=DIM_8_PIX, output_dim=DIM_8_PIX, filter_size=1, mask_type=('b', 1), inputs_stdev=np.sqrt(2), he_init=True, inputs=output)

    output = lib.ops.conv2d.Conv2D('Dec3.Out', input_dim=DIM_8_PIX, output_dim=2*DIM_8_LATENT, filter_size=1, mask_type=('b', 1), he_init=False, inputs=output)

    return output


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    all_images = tf.placeholder(tf.int32, shape=[None, N_CHANNELS, HEIGHT, WIDTH], name='all_images')

    def split(mu_and_logsig):
        mu, logsig = tf.split(1, 2, mu_and_logsig)
        # Restrict sigma to [0,1] and mu to [-5, 5]
        mu = 5. * tf.tanh(mu / 5.)
        sig = 0.5 * (tf.nn.softsign(logsig)+1)
        logsig = tf.log(sig)
        return mu, logsig, sig
 
    def clamp_logsig_and_sig(logsig, sig):
        # Early during training (see BETA_ITERS), stop sigma from going too low
        floor = 1. - tf.minimum(1., tf.cast(total_iters, 'float32') / BETA_ITERS)
        log_floor = tf.log(floor)
        return tf.maximum(logsig, log_floor), tf.maximum(sig, floor)

    split_images = tf.split(0, len(DEVICES), all_images)

    tower_cost = []

    for device, images in zip(DEVICES, split_images):
        with tf.device(device):
            embedded_images = lib.ops.embedding.Embedding('Embedding', 256, DIM_EMBED, images)
            embedded_images = tf.transpose(embedded_images, [0,4,1,2,3])
            embedded_images = tf.reshape(embedded_images, [-1, DIM_EMBED*N_CHANNELS, HEIGHT, WIDTH])

            # Layer 1
            mu_and_logsig1 = Enc1(embedded_images)
            mu1, logsig1, sig1 = split(mu_and_logsig1)

            if mu1.get_shape().as_list()[2] != LATENTS1_HEIGHT:
                raise Exception("LATENTS1_HEIGHT doesn't match mu1 shape!")
            if mu1.get_shape().as_list()[3] != LATENTS1_WIDTH:
                raise Exception("LATENTS1_WIDTH doesn't match mu1 shape!")

            eps = tf.random_normal(tf.shape(mu1))
            latents1 = mu1 + (eps * sig1)

            outputs1 = Dec1(latents1, embedded_images)

            reconst_cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    tf.reshape(outputs1, [-1, 256]),
                    tf.reshape(images, [-1])
                )
            )

            # Layer 2

            # No need to inject noise into the encoder, so I pass mu1
            # instead of latents1 to Enc2
            mu_and_logsig2 = Enc2(mu1)
            mu2, logsig2, sig2 = split(mu_and_logsig2)

            eps = tf.random_normal(tf.shape(mu2))
            latents2 = mu2 + (eps * sig2)

            outputs2 = Dec2(latents2, latents1)

            mu1_prior, logsig1_prior, sig1_prior = split(outputs2)
            logsig1_prior, sig1_prior = clamp_logsig_and_sig(logsig1_prior, sig1_prior)

            # Layer 3

            # No need to inject noise into the encoder, so I pass mu1
            # instead of latents1 to Enc2
            mu_and_logsig3 = Enc3(mu2)
            mu3, logsig3, sig3 = split(mu_and_logsig3)

            eps = tf.random_normal(tf.shape(mu3))
            latents3 = mu3 + (eps * sig3)

            outputs3 = Dec3(latents3, latents2)

            mu2_prior, logsig2_prior, sig2_prior = split(outputs3)
            logsig2_prior, sig2_prior = clamp_logsig_and_sig(logsig2_prior, sig2_prior)

            # Assembly

            # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
            # careful to avoid it.
            alpha1 = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA1_ITERS) * KL_PENALTY
            alpha2 = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA2_ITERS) * alpha1
            alpha3 = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA3_ITERS) * alpha2

            kl_cost_1 = tf.reduce_mean(
                lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
                    mu1, 
                    logsig1,
                    sig1,
                    mu1_prior,
                    logsig1_prior,
                    sig1_prior
                )
            )

            kl_cost_2 = tf.reduce_mean(
                lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
                    mu2, 
                    logsig2,
                    sig2,
                    mu2_prior,
                    logsig2_prior,
                    sig2_prior
                )
            )

            kl_cost_3 = tf.reduce_mean(
                lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                    mu3, 
                    logsig3,
                    sig3
                )
            )

            kl_cost_1 *= float(DIM_64_LATENT * LATENTS1_WIDTH * LATENTS1_HEIGHT) / (N_CHANNELS * WIDTH * HEIGHT)
            kl_cost_2 *= float(DIM_8_LATENT * LATENTS2_WIDTH * LATENTS2_HEIGHT) / (N_CHANNELS * WIDTH * HEIGHT)
            kl_cost_3 *= float(DIM_1_LATENT) / (N_CHANNELS * WIDTH * HEIGHT)

            cost = reconst_cost + (alpha1 * kl_cost_1) + (alpha2 * kl_cost_2) + (alpha3 * kl_cost_3)

            tower_cost.append(cost)

    full_cost = tf.reduce_mean(
        tf.concat(0, [tf.expand_dims(x, 0) for x in tower_cost]), 0
    )

    # Train!

    prints=[
        ('alpha1', alpha1),
        ('alpha2', alpha2),
        ('alpha3', alpha3),
        ('reconst', reconst_cost), 
        ('kl1', kl_cost_1),
        ('kl2', kl_cost_2),
        ('kl3', kl_cost_3)
    ]

    decayed_lr = tf.train.exponential_decay(
        LR,
        total_iters,
        LR_DECAY_AFTER,
        LR_DECAY_FACTOR,
        staircase=True
    )

    lib.train_loop_2.train_loop(
        session=session,
        inputs=[total_iters, all_images],
        inject_iteration=True,
        cost=full_cost,
        stop_after=TIMES['stop_after'],
        prints=prints,
        optimizer=tf.train.AdamOptimizer(decayed_lr),
        train_data=train_data,
    )