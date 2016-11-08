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

import tflib.mnist
import tflib.save_images

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

BATCH_SIZE = 100
ITERS = 1000

def LeakyReLU(x, alpha):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs, alpha=0.):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output, alpha=alpha)

def Generator(n_samples):
    noise = tf.random_uniform(
        shape=[n_samples, 100], 
        minval=-np.sqrt(3),
        maxval=np.sqrt(3)
    )

    output = ReLULayer('Generator.1', 100, 1200, noise)
    output = ReLULayer('Generator.2', 1200, 1200, output)
    
    return tf.nn.sigmoid(
        lib.ops.linear.Linear('Generator.5', 1200, 784, output)
    )

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 784, 240, inputs, alpha=0.2)
    output = ReLULayer('Discriminator.2', 240, 240, output, alpha=0.2)
    # We apply the sigmoid in a later step
    return lib.ops.linear.Linear('Discriminator.Output', 240, 1, output)#.flatten()

real_images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 784])

fake_images = Generator(BATCH_SIZE)

disc_real = Discriminator(real_images) 
disc_fake = Discriminator(fake_images)

# Gen objective:  push D(fake) to one
gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))

# Discrim objective: push D(fake) to zero, and push D(real) to one
disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
disc_cost /= 2.

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

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
    for iteration in xrange(ITERS):
        _images, _targets = gen.next()
        _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_images:_images})
        _gen_cost, _ = session.run([gen_cost, gen_train_op])

        if iteration % 100 == 0:
            print "{}\t{}\t{}".format(iteration, _disc_cost, _gen_cost)
            generate_samples(iteration)

    print "Inception score: {}".format(
        inception_score.InceptionScore().score(session.run(Generator(10000)))
    )