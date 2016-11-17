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

import tflib.enwik8
import tflib.save_images

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

BATCH_SIZE = 100
ITERS = 10000
SEQ_LEN = 1

train_data, dev_data, test_data, charmap, inv_charmap = lib.enwik8.load(BATCH_SIZE, SEQ_LEN, SEQ_LEN, 0)

lib.ops.linear.enable_default_weightnorm()

def LeakyReLU(x, alpha):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs, alpha=0.):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output, alpha=alpha)

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
        shape=[n_samples, SEQ_LEN, 128], 
        minval=-np.sqrt(3),
        maxval=np.sqrt(3)
    )

    output = lib.ops.gru.GRU('Generator.1', 128, 512, noise)
    output = lib.ops.gru.GRU('Generator.2', 512, 512, output)
    output = lib.ops.linear.Linear('Generator.Out', 512, len(charmap), output)
    return tf.reshape(tf.nn.softmax(tf.reshape(output, [-1, len(charmap)])), [n_samples, SEQ_LEN, len(charmap)])

def Discriminator(inputs):
    output = inputs
    output = lib.ops.gru.GRU('Discriminator.GRU', len(charmap), 512, output)
    output = output[:, SEQ_LEN-1, :] # last hidden state
    output = MinibatchLayer('Discriminator.Minibatch', 512, 32, 16, output)
    output = ReLULayer('Discriminator.FC', 512+32, 512, output)
    output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)
    return output # we apply the sigmoid later

real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = Generator(BATCH_SIZE)

disc_real = Discriminator(real_inputs) 
disc_fake = Discriminator(fake_inputs)

# Gen objective:  push D(fake) to one
gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))

# Discrim objective: push D(fake) to zero, and push D(real) to one
disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
disc_cost /= 2.

gen_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
disc_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

def inf_train_gen():
    while True:
        for data in train_data():
            yield data

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    def generate_samples(iteration):
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in xrange(len(samples)):
            decoded = ""
            for j in xrange(len(samples[i])):
                decoded += inv_charmap[samples[i][j]]
            decoded_samples.append(decoded)
        with open('samples_{}.txt'.format(iteration), 'w') as f:
            for s in decoded_samples:
                f.write(s + "\n")

    gen = inf_train_gen()
    gen_costs = []
    disc_costs = []
    for iteration in xrange(ITERS):

        _inputs, _targets = gen.next()

        start_time = time.time()
        _disc_cost, _gen_cost, _, _ = session.run([disc_cost, gen_cost, disc_train_op, gen_train_op], feed_dict={real_inputs_discrete:_inputs})
        end_time = time.time() - start_time

        gen_costs.append(_gen_cost)
        disc_costs.append(_disc_cost)

        if iteration % 100 == 0:
            print "iter:{}\tdisc:{:.3f}\tgen:{:.3f}\ttime:{:.3f}".format(
                iteration, 
                np.mean(disc_costs), 
                np.mean(gen_costs), 
                end_time,
            )
            gen_costs = []
            disc_costs = []
            generate_samples(iteration)
