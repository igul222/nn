"""Generative Adversarial Network for MNIST."""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True)
except ImportError:
    pass

import inception_score

import tflib as lib
import tflib.debug
import tflib.ops.linear
import tflib.ops.rnn
import tflib.ops.gru

import tflib.mnist
import tflib.save_images

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

BATCH_SIZE = 100
ITERS = 10000

lib.ops.linear.enable_default_weightnorm()

def LeakyReLU(x, alpha):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs, alpha=0.):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output, alpha=alpha)

def MiniminibatchLayer(name, n_in, dim_b, dim_c, group_size, inputs):
    inputs = tf.random_shuffle(inputs)
    inputs = tf.reshape(inputs, [-1, group_size, n_in])
    def f(a,x):
        return MinibatchLayer(name, n_in, dim_b, dim_c, x)
    outputs = tf.scan(f, inputs)
    return tf.reshape(outputs, [-1, n_in+dim_b])

def MinibatchLayer(name, n_in, dim_b, dim_c, inputs):
    """Salimans et al. 2016"""
    # input: batch_size, n_in
    # M: batch_size, dim_b, dim_c
    m = lib.ops.linear.Linear(name+'.M', n_in, dim_b*dim_c, inputs)
    m = tf.reshape(m, [-1, dim_b, dim_c])
    # c: batch_size, batch_size, dim_b
    c = tf.abs(tf.expand_dims(m, 0) - tf.expand_dims(m, 1))
    c = tf.reduce_sum(c, reduction_indices=[3])
    c = tf.exp(-c)
    # o: batch_size, dim_b
    o = tf.reduce_mean(c, reduction_indices=[1])
    o -= 1 # to account for the zero L1 distance of each example with itself
    # result: batch_size, n_in+dim_b
    return tf.concat(1, [o, inputs])

def Generator(n_samples):
    noise = tf.random_uniform(
        shape=[n_samples, 784, 32], 
        minval=-np.sqrt(3),
        maxval=np.sqrt(3)
    )

    output = lib.ops.gru.GRU('Generator.1', 32, 256, noise)
    # output = lib.ops.gru.GRU('Generator.2', 256, 256, output)
    output = lib.ops.linear.Linear('Generator.Out', 256, 1, output)
    return tf.nn.sigmoid(tf.reshape(output, [-1, 784]))

    # output = ReLULayer('Generator.1', 100, 1024, noise)
    # output = ReLULayer('Generator.2', 1024, 1024, output)
    # output = ReLULayer('Generator.2', 1024, 1024, output)
    
    # return tf.nn.sigmoid(
    #     lib.ops.linear.Linear('Generator.5', 1024, 784, output)
    # )

def PreDiscriminator(inputs):
    outputs = []
    for n_rows in [784]:
        output = tf.reshape(inputs, [-1, n_rows, 1])
        output = tf.gather(output, tf.random_shuffle(tf.range((784/n_rows)*BATCH_SIZE))[:BATCH_SIZE])
        output = lib.ops.gru.GRU('Discriminator.GRU_{}'.format(1), 1, 256, output)
        outputs.append(output)
    return outputs

def Discriminator(inputs):
    costs_ones = []
    costs_zeros = []
    outputs = []
    for n_rows in [784]:
        output = tf.reshape(inputs, [-1, n_rows, 1])
        output = tf.gather(output, tf.random_shuffle(tf.range((784/n_rows)*BATCH_SIZE))[:BATCH_SIZE])
        output = lib.ops.gru.GRU('Discriminator.GRU_{}'.format(1), 1, 256, output)
        # output = lib.ops.gru.GRU('Discriminator.GRU2', 512, 512, output)
        output = output[:, n_rows-1, :]
        output = MinibatchLayer('Discriminator.Minibatch_{}'.format(1), 256, 32, 16, output)
        output = ReLULayer('Discriminator.FC_{}'.format(1), 256+32, 256, output)
        output = lib.ops.linear.Linear('Discriminator.Output_{}'.format(1), 256, 1, output)
        cost_ones = tf.nn.sigmoid_cross_entropy_with_logits(output, tf.ones_like(output))
        cost_zeros = tf.nn.sigmoid_cross_entropy_with_logits(output, tf.zeros_like(output))
        costs_ones.append(cost_ones)
        costs_zeros.append(cost_zeros)
        outputs.append(tf.nn.sigmoid(output))
    return costs_ones, costs_zeros, outputs

real_images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 784])

fake_images = Generator(BATCH_SIZE)

disc_real_ones, disc_real_zeros, disc_reals = Discriminator(real_images) 
disc_fake_ones, disc_fake_zeros, disc_fakes = Discriminator(fake_images)

# Gen objective:  push D(fake) to one
# gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
gcs = disc_fake_ones
# ggs = [tf.global_norm(tf.gradients(gc, lib.params_with_name('Generator'))) for gc in gcs]
gen_cost = tf.reduce_mean(tf.concat(0, disc_fake_ones))

# Discrim objective: push D(fake) to zero, and push D(real) to one
# disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
# disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
dcs = [(x+y)/2. for x,y in zip(disc_fake_zeros,disc_real_ones)]
disc_cost = tf.reduce_mean(tf.concat(0, disc_fake_zeros + disc_real_ones))

gen_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
disc_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

train_data, dev_data, test_data = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for data in train_data():
            yield data

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    def generate_samples(iteration):
        samples = session.run(fake_images)
        lib.save_images.save_images(samples.reshape((-1,28,28)), 'samples_{}.jpg'.format(iteration))

    gen = inf_train_gen()
    lib.ops.linear.disable_default_weightnorm()
    scorer = inception_score.InceptionScore()
    lib.ops.linear.enable_default_weightnorm()
    gen_costs = []
    disc_costs = []
    inceptions = []
    # gc1s = []
    # gc2s = []
    # gc3s = []
    for iteration in xrange(ITERS):
        _images, _targets = gen.next()

        start_time = time.time()
        # _images, _targets = gen.next()
        # _ = session.run(disc_train_op, feed_dict={real_images:_images})
        # _disc_cost, _gen_cost, _, _, gens, gc1, gc2, gc3 = session.run([disc_cost, gen_cost, disc_train_op, gen_train_op, fake_images] + ggs, feed_dict={real_images:_images})
        _disc_cost, _gen_cost, _, _, gens = session.run([disc_cost, gen_cost, disc_train_op, gen_train_op, fake_images], feed_dict={real_images:_images})
        end_time = time.time() - start_time

        gen_costs.append(_gen_cost)
        disc_costs.append(_disc_cost)
        inceptions.append(scorer.score(gens))
        # gc1s.append(gc1)
        # gc2s.append(gc2)
        # gc3s.append(gc3)

        if iteration % 1 == 0:
            print "iter:{}\tdisc:{:.3f}\tgen:{:.3f}\tinception:{:.3f}\ttime:{:.3f}".format(
                iteration, 
                np.mean(disc_costs), 
                np.mean(gen_costs), 
                np.mean(inceptions),
                end_time,
                # np.mean(gc1s),
                # np.mean(gc2s),
                # np.mean(gc3s)
            )
            gen_costs = []
            disc_costs = []
            inceptions = []
            # gc1s = []
            # gc2s = []
            # gc3s = []
            # generate_samples(iteration)