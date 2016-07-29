"""
Multilayer VAE + Pixel CNN
Ishaan Gulrajani + Faruk Ahmed
"""

import os, sys
sys.path.append(os.getcwd())
sys.path.append('/u/ahmedfar/Ishaan/nn/')

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True)
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
import tflib.mnist_binarized

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
import scipy.misc
from scipy.misc import imsave

import time
import functools

# Switch these lines to use 1 vs 4 GPUs
DEVICES = ['/gpu:5']
# DEVICES = ['gpu:5', '/gpu:6']
# DEVICES = ['/gpu:4', '/gpu:5', '/gpu:6', '/gpu:7']

# two_level uses Enc1/Dec1 for the bottom level, Enc2/Dec2 for the top level
# one_level uses EncFull/DecFull for the bottom (and only) level
MODE = 'one_level'        

DIM_1 = 32
DIM_2 = 32
DIM_3 = 64
DIM_4 = 64
DIM_PIX = 32

LATENT_DIM = 64

TIMES = {
    'mode': 'iters',
    'print_every': 500,
    'stop_after': 2000*500,
    'callback_every': 10000
}

ALPHA_ITERS = 1
SQUARE_ALPHA = False
BETA_ITERS = 1000

VANILLA = False

starter_learning_rate = 1e-3
cut_LR_every = 100000
cut_LR_by = 0.75

BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

PIXEL_CNN_FILTER_SIZE = 5
PIXEL_CNN_LAYERS = 6

lib.print_model_settings(locals().copy())

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

train_data, dev_data, test_data = lib.mnist_binarized.load(BATCH_SIZE, BATCH_SIZE)

def nonlinearity(x):
    return tf.nn.elu(x)

def pixcnn_gated_nonlinearity(x):
    a, b = tf.split(1, 2, x)
    return tf.sigmoid(a) * tf.tanh(b)

def EncFull(images):
    output = images
    output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=1, inputs=output, he_init=False)

    output = nonlinearity(lib.ops.conv2d.Conv2D('Enc.2', input_dim=DIM_1, output_dim=DIM_2, filter_size=3, inputs=output, stride=2))

    output = nonlinearity(lib.ops.conv2d.Conv2D('Enc.3', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))
    output = nonlinearity(lib.ops.conv2d.Conv2D('Enc.4', input_dim=DIM_2, output_dim=DIM_3, filter_size=3, inputs=output, stride=2))

    # there should be a padding from 7x7 to 8x8, but apparently tf takes care of this.

    output = nonlinearity(lib.ops.conv2d.Conv2D('Enc.5', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))
    output = nonlinearity(lib.ops.conv2d.Conv2D('Enc.6', input_dim=DIM_3, output_dim=DIM_4, filter_size=3, inputs=output, stride=2))

    output = nonlinearity(lib.ops.conv2d.Conv2D('Enc.7', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
    output = nonlinearity(lib.ops.conv2d.Conv2D('Enc.8', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

    output = tf.reshape(output, [-1, 4*4*DIM_4])
    output = lib.ops.linear.Linear('Enc.Out', input_dim=4*4*DIM_4, output_dim=2*LATENT_DIM, inputs=output)

    return output

def DecFull(latents, images):

   output = latents

   output = lib.ops.linear.Linear('Dec.Inp', input_dim=LATENT_DIM, output_dim=4*4*DIM_4, inputs=output)
   output = tf.reshape(output, [-1, DIM_4, 4, 4])

   output = nonlinearity(lib.ops.conv2d.Conv2D('Dec.1', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))
   output = nonlinearity(lib.ops.conv2d.Conv2D('Dec.2', input_dim=DIM_4, output_dim=DIM_4, filter_size=3, inputs=output))

   output = nonlinearity(lib.ops.deconv2d.Deconv2D('Dec.3', input_dim=DIM_4, output_dim=DIM_3, filter_size=3, inputs=output))
   output = nonlinearity(lib.ops.conv2d.Conv2D(    'Dec.4', input_dim=DIM_3, output_dim=DIM_3, filter_size=3, inputs=output))

   # Cut from 8x8 to 7x7
   output = tf.slice(output, [0, 0, 0, 0], [-1, -1, 7, 7])

   output = nonlinearity(lib.ops.deconv2d.Deconv2D('Dec.5', input_dim=DIM_3, output_dim=DIM_2, filter_size=3, inputs=output))
   output = nonlinearity(lib.ops.conv2d.Conv2D(    'Dec.6', input_dim=DIM_2, output_dim=DIM_2, filter_size=3, inputs=output))

   output = nonlinearity(lib.ops.deconv2d.Deconv2D('Dec.7', input_dim=DIM_2, output_dim=DIM_1, filter_size=3, inputs=output))
   output = nonlinearity(lib.ops.conv2d.Conv2D(    'Dec.8', input_dim=DIM_1, output_dim=DIM_1, filter_size=3, inputs=output))

   skip_outputs = []

   masked_images = nonlinearity(lib.ops.conv2d.Conv2D(
       'Dec.PixInp',
       input_dim=N_CHANNELS,
       output_dim=DIM_1,
       filter_size=7,
       inputs=images,
       mask_type=('a', N_CHANNELS),
       he_init=False
   ))

   output = tf.concat(1, [masked_images, output])

   for i in xrange(PIXEL_CNN_LAYERS):
       inp_dim = (2*DIM_1 if i==0 else DIM_PIX)
       output = pixcnn_gated_nonlinearity(lib.ops.conv2d.Conv2D('Dec.Pix'+str(i), input_dim=inp_dim, output_dim=2*DIM_1, filter_size=PIXEL_CNN_FILTER_SIZE, inputs=output, mask_type=('b', N_CHANNELS)))
       skip_outputs.append(output)

   output = pixcnn_gated_nonlinearity(lib.ops.conv2d.Conv2D('Dec.PixOut1', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
   skip_outputs.append(output)

   output = pixcnn_gated_nonlinearity(lib.ops.conv2d.Conv2D('Dec.PixOut2', input_dim=DIM_1, output_dim=2*DIM_1, filter_size=1, inputs=output))
   skip_outputs.append(output)

   output = lib.ops.conv2d.Conv2D('Dec.PixOut3', input_dim=DIM_PIX*len(skip_outputs), output_dim=N_CHANNELS, filter_size=1, inputs=tf.concat(1, skip_outputs), he_init=False)

   return output


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    all_images = tf.placeholder(tf.float32, shape=[None, 1, 28, 28], name='all_images')

    def split(mu_and_logsig):
        mu, logsig = tf.split(1, 2, mu_and_logsig)

        sig = tf.exp(logsig)               ### looks like the better choice based on early few iters
        # sig = tf.nn.softplus(logsig)
        # sig = tf.nn.softsign(logsig) + 1

        logsig = tf.log(sig)
        return mu, logsig, sig

    def clamp_logsig_and_sig(logsig, sig):
        floor = 1. - tf.minimum(1., tf.cast(total_iters, 'float32') / BETA_ITERS)
        log_floor = tf.log(floor)
        return tf.maximum(logsig, log_floor), tf.maximum(sig, floor)

    split_images = tf.split(0, len(DEVICES), all_images)

    tower_cost = []

    LR = tf.train.exponential_decay(starter_learning_rate, total_iters,
                                           cut_LR_every, cut_LR_by, staircase=True)

    for device, images in zip(DEVICES, split_images):
        with tf.device(device):

            scaled_images = images

            mu_and_logsig1 = EncFull(scaled_images)
            mu1, logsig1, sig1 = split(mu_and_logsig1)

            if VANILLA:
                latents1 = mu1
            else:
                eps = tf.random_normal(tf.shape(mu1))
                latents1 = mu1 + (eps * sig1)

            outputs1 = DecFull(latents1, scaled_images)

            reconst_cost = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(outputs1, images)
            )
            reconst_cost *= N_CHANNELS*HEIGHT*WIDTH
            
            # Assembly

            # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
            # careful to avoid it.
            alpha = tf.minimum(1., tf.cast(total_iters+1, 'float32') / ALPHA_ITERS)
            #alpha = tf.constant(1.)

            if SQUARE_ALPHA:
                alpha = alpha**2

            kl_cost_1 = tf.reduce_mean(
                lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                    mu1, 
                    logsig1,
                    sig1
                )
            )

            kl_cost_1 *= float(LATENT_DIM)

            if VANILLA:
                cost = reconst_cost
            else:
                cost = reconst_cost + (alpha * kl_cost_1)

            tower_cost.append(cost)

    cost = tf.reduce_mean(tf.concat(0, [tf.expand_dims(x, 0) for x in tower_cost]), 0)

    # Train!

    if MODE == 'one_level':
        prints=[
            ('alpha', alpha), 
            ('LR', LR),
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1),
        ]
    elif MODE == 'two_level':
        prints=[
            ('alpha', alpha), 
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1),
            ('kl2', kl_cost_2),
        ]

    lib.train_loop.train_loop(
        session=session,
        inputs=[total_iters, all_images],
        inject_total_iters=True,
        cost=cost,
        prints=prints,
        optimizer=tf.train.AdamOptimizer(LR),
        train_data=train_data,
        test_data=dev_data,
        # callback=generate_and_save_samples,
        times=TIMES,
        # profile=True
        debug_mode=True
    )
