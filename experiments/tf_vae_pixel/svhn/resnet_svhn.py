"""
Multilayer VAE + Pixel CNN
Ishaan Gulrajani
"""

import os, sys
if 'ISHAAN_NN_LIB' in os.environ:
    sys.path.append(os.environ['ISHAAN_NN_LIB'])
else:
    sys.path.append(os.getcwd())

N_GPUS = 1

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

# import tflib.lsun_bedrooms
# import tflib.mnist_256
# import tflib.small_imagenet
import tflib.svhn

import knn

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

# two_level uses Enc1/Dec1 for the bottom level, Enc2/Dec2 for the top level
# one_level uses EncFull/DecFull for the bottom (and only) level
MODE = 'one_level'

# Turn on/off the bottom-level PixelCNN in Dec1/DecFull
PIXEL_LEVEL_PIXCNN = True
PIXCNN_ONLY = False

# These settings are good for a 'smaller' model that trains (up to 200K iters)
# in ~1 day on a GTX 1080 (probably equivalent to 2 K40s).
DIM_EMBED = 16
N = 1
DIM_PIX_1    = 32*N
DIM_1        = 32*N
DIM_2        = 64*N
DIM_3        = 128*N
DIM_4        = 256*N
LATENT_DIM_2 = 256

ALPHA1_ITERS = 3000
# ALPHA2_ITERS = 5000
KL_PENALTY = 1.00
BETA_ITERS = 1000

# In Dec2, we break each spatial location into N blocks (analogous to channels
# in the original PixelCNN) and model each spatial location autoregressively
# as P(x)=P(x0)*P(x1|x0)*P(x2|x0,x1)... In my experiments values of N > 1
# actually hurt performance. Unsure why; might be a bug.
PIX_2_N_BLOCKS = 1

TIMES = {
    'mode': 'iters',
    'test_every': 10*1000,
    'stop_after': 200*1000,
    'callback_every': 10*1000
}

VANILLA = False
LR = 1e-3


LR_DECAY_AFTER = TIMES['stop_after']
LR_DECAY_FACTOR = 1.


BATCH_SIZE = 64
N_CHANNELS = 3
HEIGHT = 32
WIDTH = 32

train_data, test_data = lib.svhn.load(BATCH_SIZE)

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    bn_is_training = tf.placeholder(tf.bool, shape=None, name='bn_is_training')
    bn_stats_iter = tf.placeholder(tf.int32, shape=None, name='bn_stats_iter')
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    all_images = tf.placeholder(tf.int32, shape=[None, N_CHANNELS, HEIGHT, WIDTH], name='all_images')
    all_labels = tf.placeholder(tf.int32, shape=[None], name='all_labels')
    # all_latents1 = tf.placeholder(tf.float32, shape=[None, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH], name='all_latents1')

    split_images = tf.split(0, len(DEVICES), all_images)
    split_labels = tf.split(0, len(DEVICES), all_labels)
    # split_latents1 = tf.split(0, len(DEVICES), all_latents1)

    tower_cost = []
    tower_outputs1_sample = []

    for device_index, (device, images, labels) in enumerate(zip(DEVICES, split_images, split_labels)):
        with tf.device(device):

            def nonlinearity(x):
                return tf.nn.elu(x)

            def pixcnn_gated_nonlinearity(a, b):
                # return tf.sigmoid(a) * b
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
                    output = conv_1(name+'.Conv1', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init, weightnorm=False)
                    output = nonlinearity(output)
                    output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init, weightnorm=False, biases=False)
                    if device_index == 0:
                        output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2,3], output, bn_is_training, bn_stats_iter)
                    else:
                        output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2,3], output, bn_is_training, bn_stats_iter, update_moving_stats=False)
                else:
                    output = nonlinearity(output)
                    output_a = conv_1(name+'.Conv1A', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
                    output_b = conv_1(name+'.Conv1B', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
                    output = pixcnn_gated_nonlinearity(output_a, output_b)
                    output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)

                return shortcut + output

            def EncFull(images):
                output = images

                output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)


                output = ResidualBlock('EncFull.Res1Pre', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs_stdev=1,          inputs=output)
                output = ResidualBlock('EncFull.Res1', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, resample='down', inputs_stdev=1,          inputs=output)
                output = ResidualBlock('EncFull.Res2Pre', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs_stdev=np.sqrt(2), inputs=output)
                output = ResidualBlock('EncFull.Res2', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, resample='down', inputs_stdev=np.sqrt(2), inputs=output)
                output = ResidualBlock('EncFull.Res3Pre', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs_stdev=np.sqrt(3), inputs=output)
                output = ResidualBlock('EncFull.Res3', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, resample='down', inputs_stdev=np.sqrt(3), inputs=output)
                output = ResidualBlock('EncFull.Res4Pre', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None,   inputs_stdev=np.sqrt(4), inputs=output)

                output = tf.reshape(output, [-1, 4*4*DIM_4])
                output = lib.ops.linear.Linear('EncFull.ConvToFC', input_dim=4*4*DIM_4, output_dim=2*LATENT_DIM_2, initialization='glorot', inputs=output)

                return output

            def DecFull(latents, images):
                output = tf.clip_by_value(latents, -50., 50.)

                output = lib.ops.linear.Linear('DecFull.Input', input_dim=LATENT_DIM_2, output_dim=4*4*DIM_4, initialization='glorot', inputs=output)
                output = tf.reshape(output, [-1, DIM_4, 4, 4])

                output = ResidualBlock('DecFull.Res2Post', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, resample=None, inputs_stdev=np.sqrt(3), he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, resample='up', inputs_stdev=np.sqrt(3), he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res3Post', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, resample=None, inputs_stdev=np.sqrt(3), he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res4', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, resample='up', inputs_stdev=np.sqrt(4), he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res4Post', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, resample=None, inputs_stdev=np.sqrt(4), he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res5', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, resample='up', inputs_stdev=np.sqrt(5), he_init=True, inputs=output)
                output = ResidualBlock('DecFull.Res5Post', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, resample=None, inputs_stdev=np.sqrt(5), he_init=True, inputs=output)

                # position-invariant latent projection
                # output = lib.ops.linear.Linear('DecFull.Input', input_dim=LATENT_DIM_2, output_dim=DIM_1, initialization='glorot', inputs=output)
                # output = tf.tile(output, [1, HEIGHT*WIDTH])
                # output = tf.reshape(output, [-1, DIM_1, HEIGHT, WIDTH])

                if PIXEL_LEVEL_PIXCNN:

                    masked_images = lib.ops.conv2d.Conv2D('DecFull.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)

                    # Make the stdev of output and masked_images match
                    # output /= np.sqrt(6)

                    # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
                    output = tf.concat(1, [masked_images, output])

                    output = ResidualBlock('DecFull.Pix2Res', input_dim=2*DIM_1,   output_dim=DIM_PIX_1, filter_size=1, mask_type=('b', N_CHANNELS), inputs_stdev=1,          inputs=output)

                    # output = ResidualBlock('DecFull.Pix2Res', input_dim=2*DIM_1,   output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs_stdev=1,          inputs=output)
                    # output = ResidualBlock('DecFull.Pix5Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs_stdev=np.sqrt(2), inputs=output)

                    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)

                else:

                    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_1, output_dim=256*N_CHANNELS, filter_size=1, he_init=False, inputs=output)

                return tf.transpose(
                    tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
                    [0,2,3,4,1]
                )

            def split(mu_and_logsig):
                mu, logsig = tf.split(1, 2, mu_and_logsig)
                # Restrict sigma to [0,1] and mu to [-2, 2]
                # mu = 2. * tf.tanh(mu / 2.)
                sig = 0.5 * (tf.nn.softsign(logsig)+1)
                logsig = tf.log(sig)
                return mu, logsig, sig
         
            def clamp_logsig_and_sig(logsig, sig):
                # Early during training (see BETA_ITERS), stop sigma from going too low
                floor = 1. - tf.minimum(1., tf.cast(total_iters, 'float32') / BETA_ITERS)
                log_floor = tf.log(floor)
                return tf.maximum(logsig, log_floor), tf.maximum(sig, floor)


            embedded_images = lib.ops.embedding.Embedding('Embedding', 256, DIM_EMBED, images)
            embedded_images = tf.transpose(embedded_images, [0,4,1,2,3])
            embedded_images = tf.reshape(embedded_images, [-1, DIM_EMBED*N_CHANNELS, HEIGHT, WIDTH])

            if MODE == 'one_level':

                # Layer 1

                mu_and_logsig1 = EncFull(embedded_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                if VANILLA:
                    latents1 = mu1
                else:
                    eps = tf.random_normal(tf.shape(mu1))
                    latents1 = mu1 + (eps * sig1)

                outputs1 = DecFull(latents1, embedded_images)

                reconst_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        tf.reshape(outputs1, [-1, 256]),
                        tf.reshape(images, [-1])
                    )
                )

                # Assembly

                # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
                # careful to avoid it.
                alpha = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA1_ITERS) * KL_PENALTY

                kl_cost_1 = tf.reduce_mean(
                    lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                        mu1, 
                        logsig1,
                        sig1
                    )
                )

                kl_cost_1 *= float(LATENT_DIM_2) / (N_CHANNELS * WIDTH * HEIGHT)

                # Auxiliary Classifier
                classifier_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        lib.ops.linear.Linear('Classifier', input_dim=LATENT_DIM_2, output_dim=10, inputs=mu1),                        
                        # lib.ops.linear.Linear('Classifier', input_dim=LATENT_DIM_2, output_dim=10, inputs=lib.ops.batchnorm.Batchnorm('ClassifierBN', [0], mu1)),
                        labels
                    )
                )


                if VANILLA:
                    cost = reconst_cost
                else:
                    cost = reconst_cost + (alpha * kl_cost_1) + classifier_cost

            elif MODE == 'two_level':
                # Layer 1

                if EMBED_INPUTS:
                    mu_and_logsig1, h1 = Enc1(embedded_images)
                else:
                    mu_and_logsig1, h1 = Enc1(scaled_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                if mu1.get_shape().as_list()[2] != LATENTS1_HEIGHT:
                    raise Exception("LATENTS1_HEIGHT doesn't match mu1 shape!")
                if mu1.get_shape().as_list()[3] != LATENTS1_WIDTH:
                    raise Exception("LATENTS1_WIDTH doesn't match mu1 shape!")

                if VANILLA:
                    latents1 = mu1
                else:
                    eps = tf.random_normal(tf.shape(mu1))
                    latents1 = mu1 + (eps * sig1)

                if EMBED_INPUTS:
                    outputs1 = Dec1(latents1, embedded_images)
                    outputs1_sample = Dec1(latents1_sample, embedded_images)
                else:
                    outputs1 = Dec1(latents1, scaled_images)
                    outputs1_sample = Dec1(latents1_sample, scaled_images)

                reconst_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        tf.reshape(outputs1, [-1, 256]),
                        tf.reshape(images, [-1])
                    )
                )

                # Layer 2

                # No need to inject noise into the encoder, so I pass mu1
                # instead of latents1 to Enc2
                mu_and_logsig2 = Enc2(h1)
                mu2, logsig2, sig2 = split(mu_and_logsig2)

                if VANILLA:
                    latents2 = mu2
                else:
                    eps = tf.random_normal(tf.shape(mu2))
                    latents2 = mu2 + (eps * sig2)

                outputs2 = Dec2(latents2, latents1)

                mu1_prior, logsig1_prior, sig1_prior = split(outputs2)
                logsig1_prior, sig1_prior = clamp_logsig_and_sig(logsig1_prior, sig1_prior)
                mu1_prior = 2. * tf.nn.softsign(mu1_prior / 2.)

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
            # tower_outputs1_sample.append(outputs1_sample)

    full_cost = tf.reduce_mean(
        tf.concat(0, [tf.expand_dims(x, 0) for x in tower_cost]), 0
    )

    # full_outputs1_sample = tf.concat(0, tower_outputs1_sample)

    # Sampling

    if MODE == 'one_level':

        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits = tf.reshape(tf.slice(outputs1, tf.pack([0, ch_sym, y_sym, x_sym, 0]), tf.pack([-1, 1, 1, 1, -1])), [-1, 256])
        dec1_fn_out = tf.multinomial(logits, 1)[:, 0]
        def dec1_fn(_latents, _targets, _ch, _y, _x):
            return session.run(dec1_fn_out, feed_dict={latents1: _latents, images: _targets, ch_sym: _ch, y_sym: _y, x_sym: _x, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})

        # def enc_fn(_images):
        #     return session.run(latents1, feed_dict={images: _images, total_iters: 99999, is_training: False})

        sample_fn_latents1 = np.random.normal(size=(8, LATENT_DIM_2)).astype('float32')

        def generate_and_save_samples(tag):
            # SVHN KNN
            print "Running KNN"
            def extract_feats(_images):
                # return session.run(latents1, feed_dict={images: _images, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})
                return session.run(mu1, feed_dict={images: _images, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})

            knn.run_knn(extract_feats)

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

            print "Saving samples"
            color_grid_vis(
                samples, 
                8, 
                8, 
                'samples_{}.png'.format(tag)
            )


    elif MODE == 'two_level':

        def dec2_fn(_latents, _targets):
            return session.run([mu1_prior, logsig1_prior], feed_dict={latents2: _latents, latents1: _targets, total_iters: 99999, bn_is_training: False, bn_stats_iter: 0})

        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits_sym = tf.reshape(tf.slice(full_outputs1_sample, tf.pack([0, ch_sym, y_sym, x_sym, 0]), tf.pack([-1, 1, 1, 1, -1])), [-1, 256])

        def dec1_logits_fn(_latents, _targets, _ch, _y, _x):
            return session.run(logits_sym,
                               feed_dict={all_latents1: _latents,
                                          all_images: _targets,
                                          ch_sym: _ch,
                                          y_sym: _y,
                                          x_sym: _x,
                                          total_iters: 99999,
                                          bn_is_training: False, 
                                          bn_stats_iter: 0})

        N_SAMPLES = BATCH_SIZE
        if N_SAMPLES % N_GPUS != 0:
            raise Exception("N_SAMPLES must be divisible by N_GPUS")
        HOLD_Z2_CONSTANT = False
        HOLD_EPSILON_1_CONSTANT = False
        HOLD_EPSILON_PIXELS_CONSTANT = False

        # Draw z2 from N(0,I)
        z2 = np.random.normal(size=(N_SAMPLES, LATENT_DIM_2)).astype('float32')
        if HOLD_Z2_CONSTANT:
          z2[:] = z2[0][None]

        # Draw epsilon_1 from N(0,I)
        epsilon_1 = np.random.normal(size=(N_SAMPLES, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH)).astype('float32')
        if HOLD_EPSILON_1_CONSTANT:
          epsilon_1[:] = epsilon_1[0][None]

        # Draw epsilon_pixels from U[0,1]
        epsilon_pixels = np.random.uniform(size=(N_SAMPLES, N_CHANNELS, HEIGHT, WIDTH))
        if HOLD_EPSILON_PIXELS_CONSTANT:
          epsilon_pixels[:] = epsilon_pixels[0][None]


        def generate_and_save_samples(tag):
            # Draw z1 autoregressively using z2 and epsilon1
            print "Generating z1"
            z1 = np.zeros((N_SAMPLES, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH), dtype='float32')
            for y in xrange(LATENTS1_HEIGHT):
              for x in xrange(LATENTS1_WIDTH):
                z1_prior_mu, z1_prior_logsig = dec2_fn(z2, z1)
                z1[:,:,y,x] = z1_prior_mu[:,:,y,x] + np.exp(z1_prior_logsig[:,:,y,x]) * epsilon_1[:,:,y,x]

            # Draw pixels (the images) autoregressively using z1 and epsilon_x
            print "Generating pixels"
            pixels = np.zeros((N_SAMPLES, N_CHANNELS, HEIGHT, WIDTH)).astype('int32')
            for y in xrange(HEIGHT):
                for x in xrange(WIDTH):
                    for ch in xrange(N_CHANNELS):
                        # start_time = time.time()
                        logits = dec1_logits_fn(z1, pixels, ch, y, x)
                        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                        probs = probs / np.sum(probs, axis=-1, keepdims=True)
                        cdf = np.cumsum(probs, axis=-1)
                        pixels[:,ch,y,x] = np.argmax(cdf >= epsilon_pixels[:,ch,y,x,None], axis=-1)
                        # print time.time() - start_time

            # Save them
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

            print "Saving"
            rows = int(np.sqrt(N_SAMPLES))
            while N_SAMPLES % rows != 0:
                rows -= 1
            color_grid_vis(
                pixels, rows, N_SAMPLES/rows, 
                'samples_{}.png'.format(tag)
            )

    # Train!

    if MODE == 'one_level':
        prints=[
            ('alpha', alpha), 
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1),
            ('classifier', classifier_cost)
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
        LR_DECAY_AFTER,
        LR_DECAY_FACTOR,
        staircase=True
    )

    lib.train_loop_2.train_loop(
        session=session,
        inputs=[total_iters, all_images, all_labels],
        inject_iteration=True,
        bn_vars=(bn_is_training, bn_stats_iter),
        bn_stats_iters=10,
        cost=full_cost,
        stop_after=TIMES['stop_after'],
        prints=prints,
        optimizer=tf.train.AdamOptimizer(decayed_lr),
        train_data=train_data,
        test_data=test_data,
        callback=generate_and_save_samples,
        callback_every=TIMES['callback_every'],
        test_every=TIMES['test_every'],
        save_checkpoints=False
    )

    # with open('/home/ishaan/resnet_svhn_search.ndjson', 'a') as f:
    #     for entry in train_output_entries[0]:
    #         for k,v in entry.items():
    #             if isinstance(v, np.generic):
    #                 entry[k] = np.asscalar(v)
    #         f.write(json.dumps(entry) + "\n")


    # print "Loading weights"
    # np.set_printoptions(precision=5, linewidth=273, suppress=True)

    # saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    # saver.restore(session, '/home/ishaan/experiments/resnet_imagenet_seriouslybig_12_1483678340/params.ckpt')
    # def eval_fn(input_vals, is_training, i):
    #     feed_dict = {sym:real for sym, real in zip([total_iters, all_images], input_vals)}
    #     feed_dict[bn_is_training] = is_training
    #     feed_dict[bn_stats_iter] = i
    #     return session.run(
    #         [full_cost] + [p[1] for p in prints],
    #         feed_dict=feed_dict
    #     )

    # for dev_vals in dev_data():
    #     for train_vals in train_data():
    #         eval_fn([np.int32(999999)] + list(train_vals), True, 0)
    #         print eval_fn([np.int32(999999)] + list(dev_vals), False, 0)
    #     break

    # print "Train vals:"
    # train_vals = []
    # for i, input_vals in enumerate(train_data()):
    #     train_vals.append(eval_fn([np.int32(9999999)] + list(input_vals), True, i))
    #     print "{}\t{}".format(i, train_vals[-1])
    #     if i >= 999:
    #         break
    # print "Dev vals:"
    # dev_vals = []
    # for i, input_vals in enumerate(dev_data()):
    #     dev_vals.append(eval_fn([np.int32(9999999)] + list(input_vals), False, 0))
    #     print "{}\t{}\t{}\t{}".format(i, dev_vals[-1], np.mean(dev_vals, axis=0), np.std(dev_vals, axis=0))
    #     if i >= 1000:
    #         break
    # print "mean"
    # print np.mean(train_vals, axis=0)
    # print np.mean(dev_vals, axis=0)
    # print "std"
    # print np.std(train_vals, axis=0)
    # print np.std(dev_vals, axis=0)