"""Generative Adversarial Network for MNIST."""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, skip=[3])
except ImportError:
    pass

import inception_score

import tflib as lib
import tflib.debug
import tflib.ops.linear

import tflib.mnist
import tflib.save_images
import tflib.random_search

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools
import json

ITERS = 20000

configs = [
    ('gen_nonlinearity', ['relu', 'leakyrelu', 'elu']),
    ('disc_nonlinearity', ['relu', 'leakyrelu', 'elu']),
    ('disc_dim', [256, 512, 1024, 2048]),
    ('gen_dim', [256, 512, 1024, 2048]),
    ('disc_n_layers', [1,3,5]),
    ('gen_n_layers', [1,3,5]),
    ('disc_lr', [1e-4, 2e-4, 5e-4, 1e-3]),
    ('gen_lr', [1e-4, 2e-4, 5e-4, 1e-3]),
    ('disc_beta1', [0.5, 0.9]),
    ('gen_beta1', [0.5, 0.9]),
    ('disc_weightnorm', [True, False]),
    ('gen_weightnorm', [True, False]),
    ('disc_b', [16, 32, 64]),
    ('disc_c', [16, 32, 64]),
    ('batch_size', [50, 100])
]

for config in lib.random_search.random_search(configs, n_trials=-1, n_splits=3, split=2):
    print "Starting {}".format(config)

    def Layer(name, n_in, n_out, nonlinearity, weightnorm, inputs):
        output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, weightnorm=weightnorm)
        if nonlinearity=='relu':
            return tf.nn.relu(output)
        elif nonlinearity=='leakyrelu':
            return tf.maximum(0.25*output, output)
        elif nonlinearity=='elu':
            return tf.nn.elu(output)

    def MinibatchLayer(name, n_in, dim_b, dim_c, weightnorm, inputs):
        """Salimans et al. 2016"""
        # input: batch_size, n_in
        # M: batch_size, dim_b, dim_c
        m = lib.ops.linear.Linear(name+'.M', n_in, dim_b*dim_c, inputs, weightnorm=weightnorm)
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
            shape=[n_samples, 100], 
            minval=-np.sqrt(3),
            maxval=np.sqrt(3)
        )

        output = Layer('Generator.Input', 100, config['gen_dim'], config['gen_nonlinearity'], config['gen_weightnorm'], noise)
        for i in xrange(config['gen_n_layers']):
            output = Layer('Generator.{}'.format(i), config['gen_dim'], config['gen_dim'], config['gen_nonlinearity'], config['gen_weightnorm'], output)
        
        return tf.nn.sigmoid(
            lib.ops.linear.Linear('Generator.Output', config['gen_dim'], 784, output, weightnorm=config['gen_weightnorm'])
        )

    def Discriminator(inputs):
        output = Layer('Discriminator.Input', 784, config['disc_dim'], config['disc_nonlinearity'], config['disc_weightnorm'], inputs)
        for i in xrange(config['disc_n_layers']):
            output = Layer('Discriminator.{}'.format(i), config['disc_dim'], config['disc_dim'], config['disc_nonlinearity'], config['disc_weightnorm'], output)
        output = MinibatchLayer('Discriminator.Minibatch', config['disc_dim'], config['disc_b'], config['disc_c'], config['disc_weightnorm'], output)
        output = Layer('Discriminator.PreOutput', config['disc_dim']+config['disc_b'], config['disc_dim'], config['disc_nonlinearity'], config['disc_weightnorm'], output)
        # We apply the sigmoid in a later step
        return lib.ops.linear.Linear('Discriminator.Output', config['disc_dim'], 1, output, weightnorm=config['gen_weightnorm'])#.flatten()

    real_images = tf.placeholder(tf.float32, shape=[config['batch_size'], 784])

    fake_images = Generator(config['batch_size'])

    disc_real = Discriminator(real_images) 
    disc_fake = Discriminator(fake_images)

    # Gen objective:  push D(fake) to one
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))

    # Discrim objective: push D(fake) to zero, and push D(real) to one
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(learning_rate=config['gen_lr'], beta1=config['gen_beta1']).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=config['disc_lr'], beta1=config['disc_beta1']).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

    train_data, dev_data, test_data = lib.mnist.load(config['batch_size'], config['batch_size'])
    def inf_train_gen():
        while True:
            for data in train_data():
                yield data

    with tf.Session() as session:

        def generate_samples(iteration):
            samples = session.run(fake_images)
            lib.save_images.save_images(samples.reshape((-1,28,28)), 'samples_{}.jpg'.format(iteration))

        scorer = inception_score.InceptionScore()
        def calculate_inception_score():
            samples = []
            for i in xrange(10):
                samples.append(session.run(Generator(1000)))
            samples = np.concatenate(samples, axis=0)
            return scorer.score(samples)

        gen = inf_train_gen()

        session.run(tf.initialize_all_variables())

        for iteration in xrange(ITERS):
            _images, _targets = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_images:_images})
            _gen_cost, _ = session.run([gen_cost, gen_train_op])

            if iteration == 2000:
                score = calculate_inception_score()
                if score < 1.2:
                    # Everything has collapsed to a mode
                    print "score < 1.2 at 2K iters, breaking early!"
                    break

        score = calculate_inception_score()
        config['inception_score'] = float(score)
        with open('/home/ishaan/mlp_gan_results.ndjson', 'a') as f:
            f.write(json.dumps(config) + "\n")

        print "Result {}".format(config)
        lib.delete_all_params()