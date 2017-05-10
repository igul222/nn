"""
PixelVAE: A Latent Variable Model for Natural Images
Ishaan Gulrajani, Kundan Kumar, Faruk Ahmed, Adrien Ali Taiga, Francesco Visin, David Vazquez, Aaron Courville
"""

import os, sys
sys.path.append(os.getcwd())

N_GPUS = 1

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=N_GPUS)
except ImportError:
    pass

import tflib as lib
import tflib.train_loop_2
import tflib.ops.kl_unit_gaussian
import tflib.ops.kl_gaussian_gaussian
import tflib.ops.conv2d
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.ops.embedding

import tflib.lsun_bedrooms
# import tflib.mnist256_leave_digit
import tflib.small_imagenet

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

DATASET = 'lsun_32' # mnist_256, lsun_32, lsun_64, imagenet_64
SETTINGS = '32px_small' # mnist_256, 32px_small, 32px_big, 64px_small, 64px_big

# SAVEDIR = '/data/lisatmp4/faruk/pixelcnn_{}_{}'.format(DATASET, SETTINGS)
# if not os.path.exists(SAVEDIR): os.makedirs(SAVEDIR)

if SETTINGS == '32px_small':
    MODE = 'two_level'

    EMBED_INPUTS = True

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 128
    DIM_1        = 64
    DIM_2        = 128
    DIM_3        = 256
    LATENT_DIM_1 = 64
    DIM_PIX_2    = 512
    DIM_4        = 512
    LATENT_DIM_2 = 512

    ALPHA1_ITERS = 2000
    ALPHA2_ITERS = 5000
    KL_PENALTY = 1.00
    BETA_ITERS = 1000

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 1000,
        'stop_after': 200000,
        'callback_every': 1
    }

    LR = 1e-3

    LR_DECAY_AFTER = 180000
    LR_DECAY_FACTOR = 1e-1

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 32
    WIDTH = 32

    LATENTS1_HEIGHT = 8
    LATENTS1_WIDTH = 8

elif SETTINGS == '32px_big':

    MODE = 'two_level'

    EMBED_INPUTS = False

    PIXEL_LEVEL_PIXCNN = True
    HIGHER_LEVEL_PIXCNN = True

    DIM_EMBED    = 16
    DIM_PIX_1    = 256
    DIM_1        = 128
    DIM_2        = 256
    DIM_3        = 512
    LATENT_DIM_1 = 128
    DIM_PIX_2    = 512
    DIM_4        = 512
    LATENT_DIM_2 = 512

    ALPHA1_ITERS = 2000
    ALPHA2_ITERS = 5000
    KL_PENALTY = 1.00
    BETA_ITERS = 1000

    PIX_2_N_BLOCKS = 1

    TIMES = {
        'test_every': 1000,
        'stop_after': 300000,
        'callback_every': 20000
    }

    VANILLA = False
    LR = 1e-3

    LR_DECAY_AFTER = 300000
    LR_DECAY_FACTOR = 1e-1

    BATCH_SIZE = 64
    N_CHANNELS = 3
    HEIGHT = 32
    WIDTH = 32
    LATENTS1_HEIGHT = 8
    LATENTS1_WIDTH = 8

if DATASET == 'mnist_256':
    train_data, dev_data, test_data = lib.mnist_256.load(BATCH_SIZE, BATCH_SIZE)
elif DATASET == 'lsun_32':
    train_data, dev_data = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=True)
elif DATASET == 'lsun_64':
    train_data, dev_data = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=False)
elif DATASET == 'imagenet_64':
    train_data, dev_data = lib.small_imagenet.load(BATCH_SIZE)

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    bn_is_training = tf.placeholder(tf.bool, shape=None, name='bn_is_training')
    bn_stats_iter = tf.placeholder(tf.int32, shape=None, name='bn_stats_iter')
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    all_images = tf.placeholder(tf.int32, shape=[None, N_CHANNELS, HEIGHT, WIDTH], name='all_images')
    all_latents1 = tf.placeholder(tf.float32, shape=[None, LATENT_DIM_1, LATENTS1_HEIGHT, LATENTS1_WIDTH], name='all_latents1')

    split_images = tf.split(0, len(DEVICES), all_images)
    split_latents1 = tf.split(0, len(DEVICES), all_latents1)

    tower_cost = []
    tower_outputs1_sample = []

    for device_index, (device, images, latents1_sample) in enumerate(zip(DEVICES, split_images, split_latents1)):
        with tf.device(device):

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

            def ResidualBlock(name, input_dim, output_dim, inputs, filter_size, mask_type=None, resample=None, he_init=True):
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
                    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
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

            def Dec1(images):

                if WIDTH == 64:
                    if EMBED_INPUTS:
                        masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_0, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                    else:
                        masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS, output_dim=DIM_0, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                else:
                    if EMBED_INPUTS:
                        masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
                    else:
                        masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=N_CHANNELS, output_dim=DIM_1, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)

                output = masked_images

                if WIDTH == 64:
                    output = ResidualBlock('Dec1.Pix2Res', input_dim=DIM_0, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    output = ResidualBlock('Dec1.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    output = ResidualBlock('Dec1.Pix4Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                else:
                    output = ResidualBlock('Dec1.Pix2Res', input_dim=DIM_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)
                    output = ResidualBlock('Dec1.Pix3Res', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, mask_type=('b', N_CHANNELS), inputs=output)

                output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)


                return tf.transpose(
                    tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
                    [0,2,3,4,1]
                )


            scaled_images = (tf.cast(images, 'float32') - 128.) / 64.
            if EMBED_INPUTS:
                embedded_images = lib.ops.embedding.Embedding('Embedding', 256, DIM_EMBED, images)
                embedded_images = tf.transpose(embedded_images, [0,4,1,2,3])
                embedded_images = tf.reshape(embedded_images, [-1, DIM_EMBED*N_CHANNELS, HEIGHT, WIDTH])


            if EMBED_INPUTS:
                outputs1 = Dec1(embedded_images)
            else:
                outputs1 = Dec1(scaled_images)

            cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    tf.reshape(outputs1, [-1, 256]),
                    tf.reshape(images, [-1])
                )
            )

            tower_cost.append(cost)

    full_cost = tf.reduce_mean(
        tf.concat(0, [tf.expand_dims(x, 0) for x in tower_cost]), 0
    )


    # Sampling
    ch_sym = tf.placeholder(tf.int32, shape=None)
    y_sym = tf.placeholder(tf.int32, shape=None)
    x_sym = tf.placeholder(tf.int32, shape=None)
    logits = tf.reshape(tf.slice(outputs1, tf.pack([0, ch_sym, y_sym, x_sym, 0]), tf.pack([-1, 1, 1, 1, -1])), [-1, 256])
    dec1_fn_out = tf.multinomial(logits, 1)[:, 0]
    def dec1_fn(_targets, _ch, _y, _x):
        return session.run(dec1_fn_out, feed_dict={images: _targets, ch_sym: _ch, y_sym: _y, x_sym: _x, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})
    def dec1_logits_fn(_targets, _ch, _y, _x):
        return session.run(outputs1, feed_dict={images: _targets, ch_sym: _ch, y_sym: _y, x_sym: _x, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})

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

        samples = np.zeros(
            (64, N_CHANNELS, HEIGHT, WIDTH), 
            dtype='int32'
        )

        print "Generating samples"

        last_logits = None
        last_location = None
        protected_locations = []

        for y in xrange(HEIGHT):
            for x in xrange(WIDTH):
                for ch in xrange(N_CHANNELS):
                    next_sample = dec1_fn(samples, ch, y, x)
                    next_logits = dec1_logits_fn(samples, ch, y, x)

                    if last_logits is not None:
                        for ch_,y_,x_ in protected_locations:
                            if not np.allclose(next_logits[:,ch_,y_,x_], last_logits[:,ch_,y_,x_], rtol=1e-2, atol=1e-4):
                                print "Violation: ch:{},x:{},y:{} depends on future input ch:{},x:{},y:{}!".format(ch_,x_,y_,last_location[0],last_location[1],last_location[2])

                    samples[:,ch,y,x] = next_sample

                    last_location = (ch,x,y)
                    protected_locations.append((ch,y,x))
                    last_logits = next_logits

        print "Saving samples"
        color_grid_vis(
            samples, 
            8, 
            8, 
            'samples_{}.png'.format(tag)
        )


    # Train!
    prints=[
        ('reconst', cost), 
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
        bn_vars=(bn_is_training, bn_stats_iter),
        cost=full_cost,
        stop_after=TIMES['stop_after'],
        prints=prints,
        optimizer=tf.train.AdamOptimizer(decayed_lr),
        train_data=train_data,
        test_data=dev_data,
        callback=generate_and_save_samples,
        callback_every=TIMES['callback_every'],
        test_every=TIMES['test_every'],
        save_checkpoints=True,
        bn_stats_iters=10
        # SAVEDIR = SAVEDIR
    )
