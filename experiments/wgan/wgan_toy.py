"""All the cool kids are doing GANs and Ishaan was feeling left out"""

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

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools

BATCH_SIZE = 1000
ITERS = 100000

GRADIENT_LOSS = False # Standard WGAN (False) vs. my thing (True)

def ReLULayer(name, n_in, n_out, inputs, alpha=0.):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def Generator(n_samples):
    noise = tf.random_uniform(
        shape=[n_samples, 128], 
        minval=-np.sqrt(3),
        maxval=np.sqrt(3)
    )

    output = ReLULayer('Generator.1', 128, 512, noise)
    output = ReLULayer('Generator.2', 512, 512, output)
    output = lib.ops.linear.Linear('Generator.Out', 512, 2, output)

    return output

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 2, 512, inputs)
    output = ReLULayer('Discriminator.2', 512, 512, output)
    output = lib.ops.linear.Linear('Discriminator.Out', 512, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, 2])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data) 
disc_fake = Discriminator(fake_data)

# WGAN loss: disc tries to push fakes down and reals up, gen tries to push fakes up
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN gradient loss term (this is my thing)
if GRADIENT_LOSS:
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    disc_interpolates = Discriminator(interpolates)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    lipschitz_penalty = tf.reduce_mean((slopes-1)**2)
    disc_cost += 10*lipschitz_penalty
    disc_gradients = tf.reduce_mean(slopes)
else:
    # Just to make the logging code work
    disc_gradients = tf.constant(0.)

if GRADIENT_LOSS:
    # The standard WGAN settings also work, but Adam is cooler!
    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))
else:
    # Settings from WGAN paper; I haven't tried Adam  everything I've tried fails
    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

    # Build an op to do the weight clipping
    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        if '.b' not in var.name:
            print "Clipping {}".format(var.name)
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
    clip_disc_weights = tf.group(*clip_ops)

# For generating plots
frame_i = [0]
def generate_image(true_dist):
    N_POINTS = 128
    RANGE = 30

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    samples, disc_map = session.run([fake_data, (disc_real)], feed_dict={real_data:points.reshape((-1, 2))})

    plt.clf()
    plt.imshow(disc_map.reshape((N_POINTS, N_POINTS)).T[::-1, :], extent=[-RANGE, RANGE, -RANGE, RANGE], cmap='seismic', vmin=np.min(disc_map), vmax=np.max(disc_map))
    plt.colorbar()
    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+')
    plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+')
    plt.savefig('frame'+str(frame_i[0])+'.jpg')
    frame_i[0] += 1

# Dataset iterator
def inf_train_gen():
    while True:
        data = sklearn.datasets.make_swiss_roll(n_samples=BATCH_SIZE*1000, noise=0.25)[0]
        data = data.astype('float32')[:, [0, 2]]
        for i in xrange(1000):
            # yield 8.*np.random.normal(size=(BATCH_SIZE,2))
            yield data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]


# Train loop!
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    disc_costs, all_disc_gradients, gen_costs = [], [], []
    for iteration in xrange(ITERS):
        _data = gen.next()

        if iteration % 2 == 0:
            disc_iters = 100 # In theory you should be able to reduce this dramatically but let's be safe
            for i in xrange(disc_iters):
                _disc_cost, __disc_gradients, _ = session.run([disc_cost, disc_gradients, disc_train_op], feed_dict={real_data: _data})
                if not GRADIENT_LOSS:
                    _ = session.run([clip_disc_weights])
                disc_costs.append(_disc_cost)
                all_disc_gradients.append(__disc_gradients)
        else:
            if gen_train_op is not None:
                _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_data: _data})
                gen_costs.append(_gen_cost)

        if iteration % 10 == 0:
            print "iter:{}\tdisc:{:.3f} disc_gradients:{:.3f}\tgen:{:.3f}".format(iteration, np.mean(disc_costs), np.mean(all_disc_gradients), np.mean(gen_costs))
            disc_costs, all_disc_gradients, gen_costs = [], [], []

            generate_image(_data)