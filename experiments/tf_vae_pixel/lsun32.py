"""
Multilayer VAE + Pixel CNN
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

# try: # This only matters on Ishaan's computer
#     import experiment_tools
#     experiment_tools.wait_for_gpu(tf=True)
# except ImportError:
#     pass

import tflib as lib
import tflib.train_loop
import tflib.ops.kl_unit_gaussian
import tflib.ops.kl_gaussian_gaussian
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear
# import tflib.ops.softmax_and_sample

import tflib.lsun_downsampled

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

DIM_1        = 128
DIM_PIX_1    = 128
DIM_2        = 128
DIM_3        = 256
LATENT_DIM_1 = 256
DIM_PIX_2    = 512
DIM_4        = 512
DIM_5        = 2048
LATENT_DIM_2 = 2048

ALPHA_ITERS = 10000
KL2_CLAMP_ITERS = 20000
BETA_ITERS = 1000

VANILLA = False
LR = 1e-3

BATCH_SIZE = 64
N_CHANNELS = 3
HEIGHT = 32
WIDTH = 32

# DEVICES = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
DEVICES = ['/gpu:0']

TIMES = {
    'mode': 'iters',
    'print_every': 1,
    'stop_after': 10000,
    'gen_every': 1000
}

lib.print_model_settings(locals().copy())

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

train_data, dev_data = lib.lsun_downsampled.load(BATCH_SIZE)

with tf.Session() as session:

    def Enc1(inputs):
        output = inputs
        
        output = ((tf.cast(output, 'float32') / 128) - 1) * 5

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc1.1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc1.2', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, inputs=output, stride=2))

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc1.3', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc1.4', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, inputs=output, stride=2))

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc1.5', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc1.6', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

        output = lib.ops.conv2d.Conv2D('Enc1.Out', input_dim=DIM_3, output_dim=2*LATENT_DIM_1, filter_size=1, inputs=output, he_init=False)

        return output

    def Dec1(latents, images):
        latents = tf.clip_by_value(latents, -50., 50.)

        output = latents

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec1.1', input_dim=LATENT_DIM_1, output_dim=DIM_3, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec1.2', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

        output = tf.nn.relu(lib.ops.deconv2d.Deconv2D('Dec1.3', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D(    'Dec1.4', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

        output = tf.nn.relu(lib.ops.deconv2d.Deconv2D('Dec1.5', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D(    'Dec1.6', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

        images = ((tf.cast(images, 'float32') / 128) - 1) * 5

        masked_images = tf.nn.relu(lib.ops.conv2d.Conv2D(
            'Dec1.Pix1', 
            input_dim=N_CHANNELS,
            output_dim=DIM_1,
            filter_size=7, 
            inputs=images, 
            mask_type=('a', N_CHANNELS)
        ))

        # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
        output = tf.concat(3, [masked_images, output])

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec1.Pix3', input_dim=2*DIM_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec1.Pix4', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec1.Pix5', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=1, inputs=output, mask_type=('b', N_CHANNELS)))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec1.Pix6', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=1, inputs=output, mask_type=('b', N_CHANNELS)))

        output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, inputs=output, mask_type=('b', N_CHANNELS), he_init=False)

        return tf.transpose(
            tf.reshape(output, [-1, HEIGHT, WIDTH, 256, N_CHANNELS]),
            [0,1,2,4,3]
        )

    def Enc2(latents):
        latents = tf.clip_by_value(latents, -50., 50.)

        output = latents
        
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc2.1', input_dim=LATENT_DIM_1, output_dim=DIM_3, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc2.2', input_dim=DIM_3,        output_dim=DIM_4, filter_size=3, inputs=output, stride=2))

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc2.3', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Enc2.4', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

        output = tf.reshape(output, [-1, 4*4*DIM_4])

        output = tf.nn.relu(lib.ops.linear.Linear('Enc2.5', input_dim=4*4*DIM_4, output_dim=DIM_5, initialization='glorot_he', inputs=output))
        output = tf.nn.relu(lib.ops.linear.Linear('Enc2.6', input_dim=DIM_5, output_dim=DIM_5, initialization='glorot_he', inputs=output))
        output = lib.ops.linear.Linear('Enc2.7', input_dim=DIM_5, output_dim=2*LATENT_DIM_2, inputs=output)

        return output

    def Dec2(latents, targets):
        latents = tf.clip_by_value(latents, -50., 50.)

        output = latents

        output = tf.nn.relu(lib.ops.linear.Linear('Dec2.1', input_dim=LATENT_DIM_2, output_dim=DIM_5,     initialization='glorot_he', inputs=output))
        output = tf.nn.relu(lib.ops.linear.Linear('Dec2.2', input_dim=DIM_5,        output_dim=DIM_5,     initialization='glorot_he', inputs=output))
        output = tf.nn.relu(lib.ops.linear.Linear('Dec2.3', input_dim=DIM_5,      output_dim=4*4*DIM_4, initialization='glorot_he', inputs=output))

        output = tf.reshape(output, [-1, 4, 4, DIM_4])

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec2.4', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec2.5', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

        output = tf.nn.relu(lib.ops.deconv2d.Deconv2D('Dec2.6', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D(    'Dec2.7', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

        masked_targets = tf.nn.relu(lib.ops.conv2d.Conv2D(
            'Dec2.Pix1', 
            input_dim=LATENT_DIM_1,
            output_dim=DIM_3,
            filter_size=5, 
            inputs=targets, 
            mask_type=('a', 1)
        ))

        output = tf.concat(3, [masked_targets, output])

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec2.Pix3', input_dim=2*DIM_3, output_dim=DIM_PIX_2, filter_size=3, inputs=output, mask_type=('b', 1)))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec2.Pix4', input_dim=DIM_PIX_2, output_dim=DIM_PIX_2, filter_size=3, inputs=output, mask_type=('b', 1)))

        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec2.Pix7', input_dim=DIM_PIX_2, output_dim=DIM_PIX_2, filter_size=1, inputs=output, mask_type=('b', 1)))
        output = tf.nn.relu(lib.ops.conv2d.Conv2D('Dec2.Pix8', input_dim=DIM_PIX_2, output_dim=DIM_PIX_2, filter_size=1, inputs=output, mask_type=('b', 1)))

        output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=DIM_PIX_2, output_dim=2*LATENT_DIM_1, filter_size=1, inputs=output, mask_type=('b', 1), he_init=False)

        return output

    total_iters = tf.placeholder(tf.int32, shape=None)
    images = tf.placeholder(tf.int32, shape=[None, 32, 32, 3])

    def conv_split(mu_and_logsig):
        mu, logsig = tf.split(3, 2, mu_and_logsig)
        logsig = tf.log(tf.nn.softplus(logsig))
        return mu, logsig

    def fc_split(mu_and_logsig):
        mu, logsig = tf.split(1, 2, mu_and_logsig)
        logsig = tf.log(tf.nn.softplus(logsig))
        return mu, logsig

    def clamp_logsig(logsig):
        beta = tf.minimum(1., tf.cast(total_iters, 'float32') / BETA_ITERS)
        return tf.maximum(beta*logsig, logsig)

    # Layer 1

    mu_and_logsig1 = Enc1(images)
    mu1, logsig1 = conv_split(mu_and_logsig1)

    if VANILLA:
        latents1 = mu1
    else:
        eps = tf.random_normal(tf.shape(mu1))
        latents1 = mu1 + (eps * tf.exp(logsig1))

    outputs1 = Dec1(latents1, images)

    reconst_cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.reshape(outputs1, [-1, 256]),
            tf.reshape(images, [-1])
        )
    )

    # Layer 2

    mu_and_logsig2 = Enc2(latents1)
    mu2, logsig2 = fc_split(mu_and_logsig2)

    if VANILLA:
        latents2 = mu2
    else:
        eps = tf.random_normal(tf.shape(mu2))
        latents2 = mu2 + (eps * tf.exp(logsig2))

    outputs2 = Dec2(latents2, latents1)

    mu1_prior, logsig1_prior = conv_split(outputs2)
    logsig1_prior = clamp_logsig(logsig1_prior)

    # Assembly

    alpha = tf.minimum(1., tf.cast(total_iters, 'float32') / ALPHA_ITERS)
    kl2_clamp = tf.maximum(0.125, 4 * (1 - (tf.cast(total_iters, 'float32') / KL2_CLAMP_ITERS)))

    kl_cost_1 = lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
        mu1,
        logsig1,
        mu1_prior,
        logsig1_prior
    )
    kl_cost_1 = tf.reduce_mean(kl_cost_1)

    kl_cost_2 = tf.reduce_mean(
        lib.ops.kl_unit_gaussian.kl_unit_gaussian(
            mu2, 
            logsig2
        ),
        reduction_indices=[0]
    )
    clamped_kl_2 = tf.maximum(kl2_clamp, kl_cost_2)
    clamped_kl_2 = tf.reduce_mean(clamped_kl_2)
    kl_cost_2 = tf.reduce_mean(kl_cost_2)

    kl_cost_1    *= float(LATENT_DIM_1*8*8) / (N_CHANNELS * WIDTH * HEIGHT)
    kl_cost_2    *= float(LATENT_DIM_2)     / (N_CHANNELS * WIDTH * HEIGHT)
    clamped_kl_2 *= float(LATENT_DIM_2)     / (N_CHANNELS * WIDTH * HEIGHT)

    if VANILLA:
        cost = reconst_cost
    else:
        cost = reconst_cost + (alpha**2 * kl_cost_1) + (alpha**5 * clamped_kl_2)

    # # Sampling

    # dec2_fn_latents = T.matrix('dec2_fn_latents')
    # dec2_fn_targets = T.tensor4('dec2_fn_targets')
    # dec2_fn = theano.function(
    #     [dec2_fn_latents, dec2_fn_targets],
    #     split(Dec2(dec2_fn_latents, dec2_fn_targets)),
    #     on_unused_input='warn'
    # )

    # dec1_fn_latents = T.tensor4('dec1_fn_latents')
    # dec1_fn_targets = T.itensor4('dec1_fn_targets')
    # dec1_fn_ch = T.iscalar()
    # dec1_fn_y = T.iscalar()
    # dec1_fn_x = T.iscalar()
    # dec1_fn_logit = Dec1(dec1_fn_latents, dec1_fn_targets)[:, dec1_fn_ch, dec1_fn_y, dec1_fn_x]
    # dec1_fn = theano.function(
    #     [dec1_fn_latents, dec1_fn_targets, dec1_fn_ch, dec1_fn_y, dec1_fn_x],
    #     lib.ops.softmax_and_sample.softmax_and_sample(dec1_fn_logit),
    #     on_unused_input='warn'
    # )

    # enc_fn = theano.function(
    #     [images],
    #     [latents1, latents2],
    #     on_unused_input='warn'
    # )

    # sample_fn_latents2 = np.random.normal(size=(32, LATENT_DIM_2)).astype(theano.config.floatX)
    # dev_images = dev_data().next()[0]
    # sample_fn_latent1_randn = np.random.normal(size=(8,8,32,LATENT_DIM_1))

    # def generate_and_save_samples(tag):
    #     def color_grid_vis(X, nh, nw, save_path):
    #         # from github.com/Newmu
    #         X = X.transpose(0,2,3,1)
    #         h, w = X[0].shape[:2]
    #         img = np.zeros((h*nh, w*nw, 3))
    #         for n, x in enumerate(X):
    #             j = n/nw
    #             i = n%nw
    #             img[j*h:j*h+h, i*w:i*w+w, :] = x
    #         imsave(save_path, img)

    #     print "Generating latents1"

    #     r_latents1, r_latents2 = enc_fn(dev_images)
    #     sample_fn_latents2[:8:2] = r_latents2[:4]
    #     sample_fn_latents2[1::2] = sample_fn_latents2[0::2]

    #     latents1 = np.zeros(
    #         (32, LATENT_DIM_1, 8, 8),
    #         dtype='float32'
    #     )

    #     for y in xrange(8):
    #         for x in xrange(8):
    #             mu, logsig = dec2_fn(sample_fn_latents2, latents1)
    #             mu = mu[:,:,y,x]
    #             logsig = logsig[:,:,y,x]
    #             z = mu + ( np.exp(logsig) * sample_fn_latent1_randn[y,x] )
    #             latents1[:,:,y,x] = z

    #     latents1[8:16:2] = r_latents1[:4]
    #     latents1[9:17:2] = r_latents1[:4]

    #     latents1_copied = np.zeros(
    #         (64, LATENT_DIM_1, 8, 8),
    #         dtype='float32'
    #     )
    #     latents1_copied[0::2] = latents1
    #     latents1_copied[1::2] = latents1


    #     samples = np.zeros(
    #         (64, N_CHANNELS, HEIGHT, WIDTH), 
    #         dtype='int32'
    #     )

    #     print "Generating samples"
    #     for y in xrange(HEIGHT):
    #         for x in xrange(WIDTH):
    #             for ch in xrange(N_CHANNELS):
    #                 next_sample = dec1_fn(latents1_copied, samples, ch, y, x)
    #                 samples[:,ch,y,x] = next_sample

    #     samples[16:32:4] = dev_images[:4]

    #     print "Saving samples"
    #     color_grid_vis(
    #         samples, 
    #         8, 
    #         8, 
    #         'samples_{}.png'.format(tag)
    #     )

    # def generate_and_save_samples_twice(tag):
    #     generate_and_save_samples(tag)
    #     generate_and_save_samples(tag+"_2")

    # Train!

    lib.train_loop.train_loop(
        session=session,
        inputs=[total_iters, images],
        inject_total_iters=True,
        cost=cost,
        prints=[
            ('alpha', alpha), 
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1),
            ('kl2', kl_cost_2),
            ('clamped_kl2', clamped_kl_2)
        ],
        optimizer=tf.train.AdamOptimizer(LR),
        train_data=train_data,
        # test_data=dev_data,
        # callback=generate_and_save_samples,
        times=TIMES
    )
