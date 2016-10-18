"""
Train a PixCNN, then extract epsilons, then train a VAE on those.
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=1)
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

import tflib.lsun_bedrooms
import tflib.mnist_256
import tflib.small_imagenet

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

DIM_EMBED = 32
DIM_PIX = 32

STOP_AFTER = 100*500

LR = 1e-3
LR_DECAY_AFTER = STOP_AFTER
LR_DECAY_FACTOR = 1.
BATCH_SIZE = 100
N_CHANNELS = 1
HEIGHT = 28
WIDTH = 28

train_data, dev_data, test_data = lib.mnist_256.load(BATCH_SIZE, BATCH_SIZE)

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

def Pixel(images):
    output = lib.ops.conv2d.Conv2D('Pix1', input_dim=N_CHANNELS*DIM_EMBED, output_dim=DIM_PIX, filter_size=5, inputs=images, mask_type=('a', N_CHANNELS), he_init=False)
    output = ResidualBlock('Pix2Res', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=5, mask_type=('b', N_CHANNELS), inputs_stdev=1, inputs=output)
    output = ResidualBlock('Pix3Res', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=5, mask_type=('b', N_CHANNELS), inputs_stdev=1, inputs=output)
    output = ResidualBlock('Pix4Res', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=5, mask_type=('b', N_CHANNELS), inputs_stdev=1, inputs=output)
    output = ResidualBlock('Pix5Res', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=5, mask_type=('b', N_CHANNELS), inputs_stdev=1, inputs=output)
    output = ResidualBlock('Pix6Res', input_dim=DIM_PIX, output_dim=DIM_PIX, filter_size=5, mask_type=('b', N_CHANNELS), inputs_stdev=1, inputs=output)
    output = lib.ops.conv2d.Conv2D('Out', input_dim=DIM_PIX, output_dim=256*N_CHANNELS, filter_size=1, mask_type=('b', N_CHANNELS), he_init=False, inputs=output)
    return tf.transpose(
        tf.reshape(output, [-1, 256, N_CHANNELS, HEIGHT, WIDTH]),
        [0,2,3,4,1]
    )

def logits_to_epsilon_bounds(logits, images):
    probs = tf.reshape(tf.nn.softmax(tf.reshape(logits, [-1, 256])), tf.shape(logits))
    cdf_lower = tf.cumsum(probs, axis=4, exclusive=True)
    cdf_upper = tf.cumsum(probs, axis=4, exclusive=False)

    # Awful hack to select the correct values
    images_mask = tf.one_hot(images, 256)
    cdf_lower = tf.reduce_sum(cdf_lower * images_mask, reduction_indices=[4])
    cdf_upper = tf.reduce_sum(cdf_upper * images_mask, reduction_indices=[4])

    return cdf_lower, cdf_upper

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    images = tf.placeholder(tf.int32, shape=[None, N_CHANNELS, HEIGHT, WIDTH], name='images')

    embedded_images = lib.ops.embedding.Embedding('Embedding', 256, DIM_EMBED, images)
    embedded_images = tf.transpose(embedded_images, [0,4,1,2,3])
    embedded_images = tf.reshape(embedded_images, [-1, N_CHANNELS*DIM_EMBED, HEIGHT, WIDTH])

    pixel_outputs = Pixel(embedded_images)

    pixel_cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.reshape(pixel_outputs, [-1, 256]),
            tf.reshape(images, [-1])
        )
    )

    cost = pixel_cost

    # Sampling
    def color_grid_vis(X, save_path):
        n_samples = X.shape[0]
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1
        nh = rows
        nw = n_samples / rows
        # from github.com/Newmu
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
        for n, x in enumerate(X):
            j = n/nw
            i = n%nw
            img[j*h:j*h+h, i*w:i*w+w, :] = x
        imsave(save_path, img)


    ch_sym = tf.placeholder(tf.int32, shape=None)
    y_sym = tf.placeholder(tf.int32, shape=None)
    x_sym = tf.placeholder(tf.int32, shape=None)
    logits_sym = tf.reshape(tf.slice(pixel_outputs, tf.pack([0, ch_sym, y_sym, x_sym, 0]), tf.pack([-1, 1, 1, 1, -1])), [-1, 256])

    def pixel_logits_fn(_targets, _ch, _y, _x):
        return session.run(logits_sym,
                           feed_dict={images: _targets,
                                      ch_sym: _ch,
                                      y_sym: _y,
                                      x_sym: _x,
                                      total_iters: 99999})

    N_SAMPLES = 64

    # Draw epsilon from U[0,1]
    uniform_epsilon = np.random.uniform(size=(N_SAMPLES, N_CHANNELS, HEIGHT, WIDTH))

    def generate_samples(tag):
        # Draw pixels (the images) autoregressively using epsilon
        print "Generating PixelCNN samples"
        pixels = np.zeros((N_SAMPLES, N_CHANNELS, HEIGHT, WIDTH)).astype('int32')
        for y in xrange(HEIGHT):
            for x in xrange(WIDTH):
                for ch in xrange(N_CHANNELS):
                    logits = pixel_logits_fn(pixels, ch, y, x)
                    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                    probs = probs / np.sum(probs, axis=-1, keepdims=True)
                    cdf = np.cumsum(probs, axis=-1)
                    pixels[:,ch,y,x] = np.argmax(cdf >= uniform_epsilon[:,ch,y,x,None], axis=-1)

        # Save them
        print "Saving"
        color_grid_vis(pixels, 'samples_{}.png'.format(tag))

    dev_batch = dev_data().next()[0]
    def generate_epsilon_images(tag):
        eps_lower, eps_upper = session.run(
            logits_to_epsilon_bounds(pixel_outputs, images),
            feed_dict={images: dev_batch, total_iters: 99999}
        )

        # midpoint
        eps_ = (eps_lower + eps_upper) / 2.
        image = np.zeros((BATCH_SIZE, N_CHANNELS, HEIGHT, WIDTH)).astype('int32')
        for ch in xrange(N_CHANNELS):
            image[:,ch,:,:] = (eps_ * 256).astype('int32')[:,0,:,:]
        color_grid_vis(image, 'eps_mid_{}.png'.format(tag))

        # range
        eps_ = (eps_upper - eps_lower)
        for ch in xrange(N_CHANNELS):
            image[:,ch,:,:] = (eps_ * 256).astype('int32')[:,0,:,:]
        color_grid_vis(image, 'eps_range_{}.png'.format(tag))

    def generate_all(tag):
        generate_samples(tag)
        generate_epsilon_images(tag)

    decayed_lr = tf.train.exponential_decay(
        LR,
        total_iters,
        LR_DECAY_AFTER,
        LR_DECAY_FACTOR,
        staircase=True
    )

    lib.train_loop_2.train_loop(
        session=session,
        inputs=[total_iters, images],
        inject_iteration=True,
        cost=cost,
        stop_after=STOP_AFTER,
        optimizer=tf.train.AdamOptimizer(decayed_lr),
        train_data=train_data,
        test_data=dev_data,
        callback=generate_all,
        callback_every=500,
        test_every=500,
    )