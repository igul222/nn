"""
Multilayer VAE + Pixel CNN
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

N_GPUS = 2

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=N_GPUS)
except ImportError:
    pass

import tflib as lib
import tflib.debug
import tflib.train_loop
import tflib.ops.kl_unit_gaussian
import tflib.ops.kl_gaussian_gaussian
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear
import tflib.ops.batchnorm

import tflib.lsun_bedrooms

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

# two_level uses Enc1/Dec1 for the bottom level, Enc2/Dec2 for the top level
# one_level uses EncFull/DecFull for the bottom (and only) level
MODE = 'two_level'

# 32x32 or 64x64
DOWNSAMPLE = False

# Turn on/off the bottom-level PixelCNN in Dec1/DecFull
PIXEL_LEVEL_PIXCNN = True

# These settings are good for a 'smaller' model that trains (up to 200K iters)
# in ~1 day on a GTX 1080 (probably equivalent to 2 K40s).
DIM_PIX_1    = 128
DIM_1        = 64
DIM_2        = 128
DIM_3        = 256
LATENT_DIM_1 = 64
DIM_PIX_2    = 512

DIM_4        = 512
DIM_5        = 2048
LATENT_DIM_2 = 512

# In Dec2, we break each spatial location into N blocks (analogous to channels
# in the original PixelCNN) and model each spatial location autoregressively
# as P(x)=P(x0)*P(x1|x0)*P(x2|x0,x1)... In my experiments values of N > 1
# actually hurt performance. Unsure why; might be a bug.
PIX_2_N_BLOCKS = 1

VANILLA = False
LR = 1e-3

if DOWNSAMPLE:
    ALPHA1_ITERS = 5000
    ALPHA2_ITERS = 5000
    KL_PENALTY = 1.01
    BETA_ITERS = 1000

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 32
    WIDTH = 32
    LATENTS1_WIDTH = 8
    LATENTS1_HEIGHT = 8

    TIMES = {
        'mode': 'iters',
        'print_every': 1000,
        'stop_after': 200000,
        'callback_every': 2000
    }
else:
    ALPHA1_ITERS = 5000
    ALPHA2_ITERS = 20000
    KL_PENALTY = 1.01
    BETA_ITERS = 1000

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 64
    WIDTH = 64
    LATENTS1_WIDTH = 8
    LATENTS1_HEIGHT = 8
    TIMES = {
        'mode': 'iters',
        'print_every': 1000,
        'stop_after': 200000,
        'callback_every': 25000
    }

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

train_data, dev_data = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=DOWNSAMPLE)

def nonlinearity(x):
    return tf.nn.elu(x)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

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
        def conv_shortcut(*args, **kwargs):
            kwargs['output_dim'] = 4*kwargs['output_dim']
            output = lib.ops.conv2d.Conv2D(*args, **kwargs)
            output = tf.transpose(output, [0,2,3,1])
            old_shape = tf.shape(output)
            output = tf.reshape(output, tf.pack([old_shape[0], 2*old_shape[1], 2*old_shape[2], old_shape[3]/4]))
            output = tf.transpose(output, [0,3,1,2])
            return output
        conv_1        = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim, output_dim=output_dim)
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
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, mask_type=mask_type, he_init=False, biases=False, inputs=inputs)

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

    output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=5, inputs=output, he_init=False, stride=2)

    # if not DOWNSAMPLE:
        # output = ResidualBlock('Enc1.Res0', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample='down', inputs_stdev=1, inputs=output)

    output = ResidualBlock('Enc1.Res1', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, resample='down', inputs_stdev=1,          inputs=output)
    output = ResidualBlock('Enc1.Res2', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', inputs_stdev=np.sqrt(2), inputs=output)
    output = ResidualBlock('Enc1.Res3', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None,   inputs_stdev=np.sqrt(3), inputs=output)

    output = lib.ops.conv2d.Conv2D('Enc1.Out', input_dim=DIM_3, output_dim=2*LATENT_DIM_1, filter_size=1, inputs=output, he_init=False)

    return output

def Dec1(latents, images):
    output = tf.clip_by_value(latents, -50., 50.)
    output = lib.ops.conv2d.Conv2D('Dec1.Input', input_dim=LATENT_DIM_1, output_dim=DIM_3, filter_size=1, inputs=output, he_init=False)

    output = ResidualBlock('Dec1.Res1', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs_stdev=1, inputs=output)
    output = ResidualBlock('Dec1.Res2', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', inputs_stdev=np.sqrt(2), inputs=output)
    output = ResidualBlock('Dec1.Res3', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, resample='up', inputs_stdev=np.sqrt(3), inputs=output)

    if not DOWNSAMPLE:
        output = ResidualBlock('Dec1.Res4', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample='up', inputs_stdev=np.sqrt(3), inputs=output)

    if PIXEL_LEVEL_PIXCNN:

        masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=7, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)

        # Make the stdev of output and masked_images match
        output /= np.sqrt(4)

        # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
        output = tf.concat(1, [masked_images, output])

        output = ResidualBlock('Dec1.Pix2Res', input_dim=2*DIM_1,   output_dim=DIM_PIX_1, filter_size=5, mask_type=('b', N_CHANNELS), inputs_stdev=1,          inputs=output)
        output = ResidualBlock('Dec1.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=1, mask_type=('b', N_CHANNELS), inputs_stdev=np.sqrt(2), inputs=output)

        output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)

    else:

        output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_1, output_dim=256*N_CHANNELS, filter_size=1, he_init=False, inputs=output)

    return tf.transpose(
        tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
        [0,2,3,4,1]
    )

def Enc2(latents):
    output = tf.clip_by_value(latents, -50., 50.)

    output = lib.ops.conv2d.Conv2D('Enc2.Input', input_dim=LATENT_DIM_1, output_dim=DIM_3, filter_size=1, inputs=output, he_init=False)

    output = ResidualBlock('Enc2.Res1', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, resample='down', inputs_stdev=1,          he_init=True, inputs=output)
    output = ResidualBlock('Enc2.Res2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None,   inputs_stdev=np.sqrt(2), he_init=True, inputs=output)

    output = tf.reshape(output, [-1, 4*4*DIM_4])
    output = lib.ops.linear.Linear('Enc2.ConvToFC', input_dim=4*4*DIM_4, output_dim=DIM_5, initialization='glorot', inputs=output)

    # We implement an FC residual block as a conv over a 1x1 featuremap
    output = tf.reshape(output, [-1, DIM_5, 1, 1])
    output = ResidualBlock('Enc2.Res3', input_dim=DIM_5, output_dim=DIM_5, filter_size=1, inputs_stdev=np.sqrt(3), he_init=True, inputs=output)
    output = tf.reshape(output, [-1, DIM_5])

    output = lib.ops.linear.Linear('Enc2.Output', input_dim=DIM_5, output_dim=2*LATENT_DIM_2, inputs=output, initialization='glorot')
    return output

def Dec2(latents, targets):
    output = tf.clip_by_value(latents, -50., 50.)
    output = lib.ops.linear.Linear('Dec2.Input', input_dim=LATENT_DIM_2, output_dim=DIM_5, initialization='glorot', inputs=output)

    output = tf.reshape(output, [-1, DIM_5, 1, 1])
    output = ResidualBlock('Dec2.Res1', input_dim=DIM_5, output_dim=DIM_5, filter_size=1, inputs_stdev=1, he_init=True, inputs=output)
    output = tf.reshape(output, [-1, DIM_5])

    output = lib.ops.linear.Linear('Dec2.FCToConv', input_dim=DIM_5, output_dim=4*4*DIM_4, initialization='glorot', inputs=output)
    output = tf.reshape(output, [-1, DIM_4, 4, 4])

    output = ResidualBlock('Dec2.Res2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, inputs_stdev=np.sqrt(2), he_init=True, inputs=output)
    output = ResidualBlock('Dec2.Res3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, resample='up', inputs_stdev=np.sqrt(3), he_init=True, inputs=output)

    masked_targets = lib.ops.conv2d.Conv2D('Dec2.Pix1', input_dim=LATENT_DIM_1, output_dim=DIM_3, filter_size=7, mask_type=('a', PIX_2_N_BLOCKS), he_init=False, inputs=targets)

    # Make the stdev of output and masked_targets match
    output /= np.sqrt(4)

    output = tf.concat(1, [masked_targets, output])

    output = ResidualBlock('Dec2.Pix2Res', input_dim=2*DIM_3, output_dim=DIM_PIX_2, filter_size=3, mask_type=('b', PIX_2_N_BLOCKS), inputs_stdev=1, he_init=True, inputs=output)
    output = ResidualBlock('Dec2.Pix3Res', input_dim=DIM_PIX_2, output_dim=DIM_PIX_2, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), inputs_stdev=np.sqrt(2), he_init=True, inputs=output)

    output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=DIM_PIX_2, output_dim=2*LATENT_DIM_1, filter_size=1, mask_type=('b', PIX_2_N_BLOCKS), he_init=False, inputs=output)
    return output

def EncFull(images):
    output = images
    output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)

    output = ResidualBlock('EncFull.Res1', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, resample='down', inputs_stdev=1,          inputs=output)
    output = ResidualBlock('EncFull.Res2', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', inputs_stdev=np.sqrt(2), inputs=output)
    output = ResidualBlock('EncFull.Res3', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, resample='down', inputs_stdev=np.sqrt(3), inputs=output)
    output = ResidualBlock('EncFull.Res4', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None,   inputs_stdev=np.sqrt(4), inputs=output)

    output = tf.reshape(output, [-1, 4*4*DIM_4])
    output = lib.ops.linear.Linear('EncFull.ConvToFC', input_dim=4*4*DIM_4, output_dim=DIM_5, initialization='glorot', inputs=output)

    # We implement an FC residual block as a conv over a 1x1 featuremap
    output = tf.reshape(output, [-1, DIM_5, 1, 1])
    output = ResidualBlock('EncFull.Res5', input_dim=DIM_5, output_dim=DIM_5, filter_size=1, inputs_stdev=np.sqrt(5), he_init=True, inputs=output)
    output = tf.reshape(output, [-1, DIM_5])

    output = lib.ops.linear.Linear('EncFull.Output', input_dim=DIM_5, output_dim=2*LATENT_DIM_2, inputs=output, initialization='glorot')
    return output

def DecFull(latents, images):
    output = tf.clip_by_value(latents, -50., 50.)
    output = lib.ops.linear.Linear('DecFull.Input', input_dim=LATENT_DIM_2, output_dim=DIM_5, initialization='glorot', inputs=output)

    output = tf.reshape(output, [-1, DIM_5, 1, 1])
    output = ResidualBlock('DecFull.Res1', input_dim=DIM_5, output_dim=DIM_5, filter_size=1, inputs_stdev=1, he_init=True, inputs=output)
    output = tf.reshape(output, [-1, DIM_5])

    output = lib.ops.linear.Linear('DecFull.FCToConv', input_dim=DIM_5, output_dim=4*4*DIM_4, initialization='glorot', inputs=output)
    output = tf.reshape(output, [-1, DIM_4, 4, 4])

    output = ResidualBlock('DecFull.Res2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, inputs_stdev=np.sqrt(2), he_init=True, inputs=output)
    output = ResidualBlock('DecFull.Res3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, resample='up', inputs_stdev=np.sqrt(3), he_init=True, inputs=output)
    output = ResidualBlock('DecFull.Res4', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', inputs_stdev=np.sqrt(4), he_init=True, inputs=output)
    output = ResidualBlock('DecFull.Res5', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, resample='up', inputs_stdev=np.sqrt(5), he_init=True, inputs=output)

    if PIXEL_LEVEL_PIXCNN:

        masked_images = lib.ops.conv2d.Conv2D('DecFull.Pix1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)

        # Make the stdev of output and masked_images match
        output /= np.sqrt(6)

        # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
        output = tf.concat(1, [masked_images, output])

        output = ResidualBlock('Dec1.Pix2Res', input_dim=2*DIM_1,   output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs_stdev=1,          inputs=output)
        output = ResidualBlock('Dec1.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=1, mask_type=('b', N_CHANNELS), inputs_stdev=np.sqrt(2), inputs=output)

        output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)

    else:

        output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_1, output_dim=256*N_CHANNELS, filter_size=1, he_init=False, inputs=output)

    return tf.transpose(
        tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
        [0,2,3,4,1]
    )

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    all_images = tf.placeholder(tf.int32, shape=[None, N_CHANNELS, HEIGHT, WIDTH], name='all_images')

    def split(mu_and_logsig):
        mu, logsig = tf.split(1, 2, mu_and_logsig)
        sig = tf.nn.softsign(logsig)+1
        logsig = tf.log(sig)
        return mu, logsig, sig

    def clamp_logsig_and_sig(logsig, sig):
        floor = 1. - tf.minimum(1., tf.cast(total_iters, 'float32') / BETA_ITERS)
        log_floor = tf.log(floor)
        return tf.maximum(logsig, log_floor), tf.maximum(sig, floor)

    split_images = tf.split(0, len(DEVICES), all_images)

    tower_cost = []

    for device, images in zip(DEVICES, split_images):
        with tf.device(device):

            scaled_images = (tf.cast(images, 'float32') - 128.) / 64.

            if MODE == 'one_level':

                # Layer 1

                mu_and_logsig1 = EncFull(scaled_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                if VANILLA:
                    latents1 = mu1
                else:
                    eps = tf.random_normal(tf.shape(mu1))
                    latents1 = mu1 + (eps * sig1)

                outputs1 = DecFull(latents1, scaled_images)

                reconst_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        tf.reshape(outputs1, [-1, 256]),
                        tf.reshape(images, [-1])
                    )
                )

                # Assembly

                # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
                # careful to avoid it.
                alpha = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA_ITERS)
                
                kl_cost_1 = tf.reduce_mean(
                    lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                        mu1, 
                        logsig1,
                        sig1
                    )
                )

                kl_cost_1 *= float(LATENT_DIM_2) / (N_CHANNELS * WIDTH * HEIGHT)

                if VANILLA:
                    cost = reconst_cost
                else:
                    cost = reconst_cost + (alpha * kl_cost_1)

            elif MODE == 'two_level':
                # Layer 1

                mu_and_logsig1 = Enc1(scaled_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                if VANILLA:
                    latents1 = mu1
                else:
                    eps = tf.random_normal(tf.shape(mu1))
                    latents1 = mu1 + (eps * sig1)

                outputs1 = Dec1(latents1, scaled_images)

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

                if VANILLA:
                    latents2 = mu2
                else:
                    eps = tf.random_normal(tf.shape(mu2))
                    latents2 = mu2 + (eps * sig2)

                outputs2 = Dec2(latents2, latents1)

                mu1_prior, logsig1_prior, sig1_prior = split(outputs2)
                logsig1_prior, sig1_prior = clamp_logsig_and_sig(logsig1_prior, sig1_prior)

                # Assembly

                # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
                # careful to avoid it.
                alpha1 = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA1_ITERS) * KL_PENALTY
                alpha2 = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA2_ITERS) * alpha1# * KL_PENALTY

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
                    lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                        mu2, 
                        logsig2,
                        sig2
                    )
                )

                kl_cost_1 *= float(LATENT_DIM_1 * LATENTS1_WIDTH * LATENTS1_HEIGHT) / (N_CHANNELS * WIDTH * HEIGHT)
                kl_cost_2 *= float(LATENT_DIM_2) / (N_CHANNELS * WIDTH * HEIGHT)

                if VANILLA:
                    cost = reconst_cost
                else:
                    cost = reconst_cost + (alpha1 * kl_cost_1) + (alpha2 * kl_cost_2)

            tower_cost.append(cost)

    full_cost = tf.reduce_mean(
        tf.concat(0, [tf.expand_dims(x, 0) for x in tower_cost]), 0
    )

    # Sampling

    if MODE == 'one_level':

        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits = tf.reshape(tf.slice(outputs1, tf.pack([0, ch_sym, y_sym, x_sym, 0]), tf.pack([-1, 1, 1, 1, -1])), [-1, 256])
        dec1_fn_out = tf.multinomial(logits, 1)[:, 0]
        def dec1_fn(_latents, _targets, _ch, _y, _x):
            return session.run(dec1_fn_out, feed_dict={latents1: _latents, images: _targets, ch_sym: _ch, y_sym: _y, x_sym: _x, total_iters: 99999})

        def enc_fn(_images):
            return session.run(latents1, feed_dict={images: _images, total_iters: 99999})

        sample_fn_latents1 = np.random.normal(size=(8, LATENT_DIM_2)).astype('float32')
        dev_images = dev_data().next()[0]

        def generate_and_save_samples(tag):
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

            print "Generating latents1"

            r_latents1 = enc_fn(dev_images)
            for i in xrange(4):
                sample_fn_latents1[i] = r_latents1[i]

            latents1_copied = np.zeros((64, LATENT_DIM_2), dtype='float32')
            for i in xrange(8):
                latents1_copied[i::8] = sample_fn_latents1

            samples = np.zeros(
                (64, N_CHANNELS, HEIGHT, WIDTH), 
                dtype='int32'
            )

            print "Generating samples"
            for y in xrange(HEIGHT):
                for x in xrange(WIDTH):
                    for ch in xrange(N_CHANNELS):
                        next_sample = dec1_fn(latents1_copied, samples, ch, y, x)
                        samples[:,ch,y,x] = next_sample

            samples[:32:8] = dev_images[:4]

            print "Saving samples"
            color_grid_vis(
                samples, 
                8, 
                8, 
                'samples_{}.png'.format(tag)
            )


    elif MODE == 'two_level':

        def dec2_fn(_latents, _targets):
            return session.run([mu1_prior, logsig1_prior], feed_dict={latents2: _latents, latents1: _targets, total_iters: 99999})

        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits = tf.reshape(tf.slice(outputs1, tf.pack([0, ch_sym, y_sym, x_sym, 0]), tf.pack([-1, 1, 1, 1, -1])), [-1, 256])
        dec1_fn_out = tf.multinomial(logits, 1)[:, 0]
        def dec1_fn(_latents, _targets, _ch, _y, _x):
            return session.run(dec1_fn_out, feed_dict={latents1: _latents, images: _targets, ch_sym: _ch, y_sym: _y, x_sym: _x, total_iters: 99999})

        def enc_fn(_images):
            return session.run([latents1, latents2], feed_dict={images: _images, total_iters: 99999})

        sample_fn_latents2 = np.random.normal(size=(32, LATENT_DIM_2)).astype('float32')
        sample_fn_latents2[1::2] = sample_fn_latents2[0::2]
        sample_fn_latent1_randn = np.random.normal(size=(LATENTS1_HEIGHT,LATENTS1_WIDTH,32,LATENT_DIM_1))

        def generate_and_save_samples(tag):
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

            print "Generating latents1"

            latents1 = np.zeros(
                (32, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH),
                dtype='float32'
            )

            for y in xrange(8):
                for x in xrange(8):
                    for block in xrange(PIX_2_N_BLOCKS):
                        mu, logsig = dec2_fn(sample_fn_latents2, latents1)
                        mu = mu[:,:,y,x]
                        logsig = logsig[:,:,y,x]
                        z = mu + ( np.exp(logsig) * sample_fn_latent1_randn[y,x] )
                        latents1[:,block::PIX_2_N_BLOCKS,y,x] = z[:,block::PIX_2_N_BLOCKS]

            latents1_copied = np.zeros(
                (64, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH),
                dtype='float32'
            )
            latents1_copied[0::2] = latents1
            latents1_copied[1::2] = latents1

            samples = np.zeros(
                (64, N_CHANNELS, HEIGHT, WIDTH), 
                dtype='int32'
            )

            print "Generating samples"
            for y in xrange(HEIGHT):
                for x in xrange(WIDTH):
                    for ch in xrange(N_CHANNELS):
                        next_sample = dec1_fn(latents1_copied, samples, ch, y, x)
                        samples[:,ch,y,x] = next_sample

            print "Saving samples"
            color_grid_vis(
                samples,
                8,
                8,
                'samples_{}.png'.format(tag)
            )

    # Train!

    if MODE == 'one_level':
        prints=[
            ('alpha', alpha), 
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1)
        ]
    elif MODE == 'two_level':
        prints=[
            ('alpha1', alpha1),
            ('alpha2', alpha2),
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1),
            ('kl2', kl_cost_2),
        ]

    decayed_lr = tf.train.exponential_decay(
        LR,
        total_iters,
        TIMES['stop_after']-10000,
        1e-1,
        staircase=True
    )

    lib.train_loop.train_loop(
        session=session,
        inputs=[total_iters, all_images],
        inject_total_iters=True,
        cost=full_cost,
        prints=prints,
        optimizer=tf.train.AdamOptimizer(decayed_lr),
        train_data=train_data,
        # test_data=dev_data,
        callback=generate_and_save_samples,
        times=TIMES,
        # profile=True
        # debug_mode=True
    )