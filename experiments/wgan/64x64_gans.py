"""Lots of GAN architectures on LSUN bedrooms."""

import os, sys
sys.path.append(os.getcwd())

N_GPUS = 2

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=N_GPUS, skip=[1])
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
import tflib.small_imagenet
import tflib.ops.gru

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools
import collections

BATCH_SIZE = 64
ITERS = 50000
MODE = 'dcgan' # dcgan, wgan, wgan++, lsgan
DATASET = 'imagenet' # imagenet, lsun

def GeneratorAndDiscriminator():
    return functools.partial(DCGANGenerator, dim=64, bn=True), functools.partial(DCGANDiscriminator, dim=64, bn=True)
    # return WGANPaper_CrippledDCGANGenerator, DCGANDiscriminator

OUTPUT_DIM = 64*64*3

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

# ! Layers

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='glorot')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='glorot')
    return LeakyReLU(output)

def Batchnorm(name, axes, inputs):
    # Switch batchnorm with layer norm in WGAN++.
    # Also, TF's fused batchnorm implementation doesn't have gradients-of-gradients.
    if ('Discriminator' in name) and (MODE == 'wgan++'):
        return inputs
        for i in xrange(len(axes)):
            if axes[i] == 0:
                axes[i] = 1
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=False)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs)

# ! Generators

def FCGenerator(n_samples, noise=None, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, FC_DIM, noise)
    output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Generator.Out', FC_DIM, OUTPUT_DIM, output)

    output = tf.tanh(output)

    return output

def DCGANGenerator(n_samples, noise=None, dim=128, bn=True):
    # lib.ops.conv2d.set_weights_stdev(0.02)
    # lib.ops.deconv2d.set_weights_stdev(0.02)
    # lib.ops.linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    if bn:
        output = Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    # lib.ops.conv2d.unset_weights_stdev()
    # lib.ops.deconv2d.unset_weights_stdev()
    # lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM])

def WGANPaper_CrippledDCGANGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    dim = 128

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*dim, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, dim, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


# ! Discriminators

def FCDiscriminator(inputs, FC_DIM=512):
    output = LeakyReLULayer('Discriminator.1', OUTPUT_DIM, FC_DIM, inputs)
    output = LeakyReLULayer('Discriminator.2', FC_DIM, FC_DIM, output)
    output = LeakyReLULayer('Discriminator.3', FC_DIM, FC_DIM, output)
    output = LeakyReLULayer('Discriminator.4', FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])

def DCGANDiscriminator(inputs, dim=128, bn=True):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1])

Generator, Discriminator = GeneratorAndDiscriminator()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs = [],[]

    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):

            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE/len(DEVICES))

            # TODO fix
            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            if MODE == 'wgan':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
                mean_grad_norm = tf.constant(0.) # to make the logging code work

            elif MODE == 'wgan++':
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
                lipschitz_penalty = tf.reduce_mean((slopes-1.)**2)
                wgan_disc_cost = disc_cost
                disc_cost += 10*lipschitz_penalty

            elif MODE == 'dcgan':
                gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
                disc_cost /= 2.
                mean_grad_norm = tf.constant(0.) # to make the logging code work

            elif MODE == 'lsgan':
                gen_cost = tf.reduce_mean((disc_fake - 1)**2)
                disc_cost = (tf.reduce_mean((disc_real - 1)**2) + tf.reduce_mean((disc_fake - 0)**2))/2.
                mean_grad_norm = tf.constant(0.) # to make the logging code work


            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    if MODE == 'wgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        clip_ops = []
        for var in lib.params_with_name('Discriminator'):
            print "Clipping {}".format(var.name)
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan++':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'dcgan':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'lsgan':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)


    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE / len(DEVICES)
        all_fixed_noise_samples.append(Generator(n_samples, noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples]))
    all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)
    def generate_image(frame, true_dist):
        samples = session.run(all_fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'samples_{}.jpg'.format(frame))


    if DATASET == 'lsun':
        train_gen, _ = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=False)
    elif DATASET == 'imagenet':
        train_gen, _ = lib.small_imagenet.load(BATCH_SIZE)

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images


    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    _disc_costs, _gen_costs, times = [], [], []


    for iteration in xrange(ITERS):


        start_time = time.time()

        if (MODE == 'dcgan') or (MODE == 'lsgan'):
            disc_iters = 1
        else:
            if iteration < 20:
                disc_iters = 20
            else:
                disc_iters = 5
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})
            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])

        _disc_costs.append(_disc_cost)

        _data = gen.next()
        _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={all_real_data_conv: _data})
        _gen_costs.append(_gen_cost)

        times.append(time.time() - start_time)

        if (iteration < 20) or (iteration % 20 == 19):
            print "iter:\t{}\tdisc:\t{:.3f}\tgen:\t{:.3f}\ttime:\t{:.3f}".format(iteration, np.mean(_disc_costs), np.mean(_gen_costs), np.mean(times))
            _disc_costs, _gen_costs, times = [], [], []

        if iteration % 100 == 0:
            generate_image(iteration, _data)