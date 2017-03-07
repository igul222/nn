"""Generative Adversarial Network for MNIST."""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True)
except ImportError:
    pass

import tflib as lib
import tflib.debug
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.adamax
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.lsun_bedrooms

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools

BATCH_SIZE = 128
ITERS = 100000
# DIM = 32
# DIM_G = 32
DATASET = 'lsun' # mnist, lsun
MODE = 'dcgan' # dcgan, wgan, igan

if DATASET == 'mnist':
    OUTPUT_DIM = 28*28
elif DATASET == 'lsun':
    OUTPUT_DIM = 64*64*3

if MODE == 'dcgan':
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def FCGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, 512, noise)
    output = ReLULayer('Generator.2', 512, 512, output)
    output = ReLULayer('Generator.3', 512, 512, output)
    output = lib.ops.linear.Linear('Generator.Out', 512, OUTPUT_DIM, output)

    output = tf.nn.sigmoid(output)

    return output

def FCDiscriminator(inputs):

    output = LeakyReLULayer('Discriminator.1', OUTPUT_DIM, 512, inputs)
    output = LeakyReLULayer('Discriminator.2', 512, 512, output)
    output = LeakyReLULayer('Discriminator.3', 512, 512, output)
    output = lib.ops.linear.Linear('Discriminator.Out', OUTPUT_DIM, 1, output)

    return tf.reshape(output, [-1])

def DCGANGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*1024, noise)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 1024, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 1024, 512, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 512, 256, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 256, 128, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', 128, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def DCGANDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, 128, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', 128, 256, 5, output, stride=2)
    output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 256, 512, 5, output, stride=2)
    output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 512, 1024, 5, output, stride=2)
    output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*1024])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*1024, 1, output)
    # output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1])

Generator = DCGANGenerator
Discriminator = DCGANDiscriminator

real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        if '.b' not in var.name:
            print "Clipping {}".format(var.name)
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'igan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[1,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[0,1]))
    lipschitz_penalty = tf.reduce_mean(tf.maximum(0.,slopes-10.)**2)
    wgan_disc_cost = disc_cost
    disc_cost += 10*lipschitz_penalty
    lipschitz_penalty = tf.reduce_mean(slopes)

    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))

frame_i = [0]
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((128, 3, 64, 64)), 'samples_{}.jpg'.format(frame))


if DATASET == 'mnist':
    train_gen, _, _ = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
    def inf_train_gen():
        while True:
            for images,targets in train_gen():
                yield images

elif DATASET == 'lsun':
    train_gen, _ = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=False)
    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    disc_costs, gen_costs = [], []

    start_time = time.time()

    for iteration in xrange(ITERS):

        if iteration % 2 == 0:
            if MODE == 'dcgan':
                disc_iters = 1
            else:
                if iteration < 20:
                    disc_iters = 50
                else:
                    disc_iters = 5
            for i in xrange(disc_iters):
                _data = gen.next()
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_conv: _data})

            disc_costs.append(_disc_cost)
            print _disc_cost

        else:
            _data = gen.next()
            _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_data_conv: _data})
            gen_costs.append(_gen_cost)

        if ((iteration < 20) and (iteration % 2 == 1)) or (iteration % 20 == 19):
            print "iter:\t{}\tdisc:\t{:.3f}\tgen:\t{:.3f}\ttime:\t{:.3f}".format(iteration, np.mean(disc_costs), np.mean(gen_costs), time.time() - start_time)
            disc_costs, gen_costs = [], []
            start_time = time.time()
        if iteration % 100 == 0:
            generate_image(iteration, _data)