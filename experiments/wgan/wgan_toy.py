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
import tflib.ops.batchnorm

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools

BATCH_SIZE = 200
ITERS = 100000

# MODE = 'lsgan' # dcgan, wgan, wgan++, lsgan
GRADIENT_LOSS = False
# DATASET = '25gaussians'
DATASET = 'swissroll'

debugprints = []
def debugprint(name, val):
    debugprints.append((name, val))

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs, alpha=0., bn=False, wn=False):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, weightnorm=wn)
    if bn:
        if GRADIENT_LOSS:
            output = lib.ops.batchnorm.Batchnorm(name+'.BN', [1], output)
        else:
            output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0], output)

    output = tf.nn.relu(output)
    debugprint(name, output)
    return output

def Generator(n_samples):
    noise = tf.random_normal([n_samples, 2])

    output = ReLULayer('Generator.1', 2, 512, noise)
    output = ReLULayer('Generator.2', 512, 512, output)
    output = lib.ops.linear.Linear('Generator.Out', 512, 2, output)

    return output

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 2, 512, inputs, bn=False, wn=False)
    output = ReLULayer('Discriminator.2', 512, 512, output, bn=True, wn=False)

    # If you uncomment these lines, it breaks
    output = ReLULayer('Discriminator.3', 512, 512, output, bn=True, wn=False)
    output = ReLULayer('Discriminator.4', 512, 512, output, bn=True, wn=False)
    output = ReLULayer('Discriminator.5', 512, 512, output, bn=True, wn=False)
    output = ReLULayer('Discriminator.6', 512, 512, output, bn=True, wn=False)
    output = ReLULayer('Discriminator.7', 512, 512, output, bn=True, wn=False)
    output = ReLULayer('Discriminator.8', 512, 512, output, bn=True, wn=False)
    output = ReLULayer('Discriminator.9', 512, 512, output, bn=True, wn=False)
    # output = ReLULayer('Discriminator.A', 512, 512, output, bn=True, wn=False)
    # output = ReLULayer('Discriminator.B', 512, 512, output, bn=True, wn=False)
    # output = ReLULayer('Discriminator.C', 512, 512, output, bn=True, wn=False)

    output = lib.ops.linear.Linear('Discriminator.Out', 512, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, 2])
fake_data = Generator(BATCH_SIZE)

disc_out = Discriminator(tf.concat(0, [fake_data, real_data]))
disc_fake, disc_real = disc_out[:BATCH_SIZE], disc_out[BATCH_SIZE:]
# disc_real = Discriminator(real_data) 
# disc_fake = Discriminator(fake_data)

# WGAN loss: disc tries to push fakes down and reals up, gen tries to push fakes up
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

for (name, val), grad in zip(debugprints, tf.gradients(disc_cost, [v for n,v in debugprints])):
    debugprint('grad_'+name, grad)

# WGAN gradient loss term (this is my thing)
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
if GRADIENT_LOSS:
    disc_cost += 10*lipschitz_penalty
disc_gradients = tf.reduce_mean(slopes)

if GRADIENT_LOSS:
    # The standard WGAN settings also work, but Adam is cooler!
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))
else:
    # Settings from WGAN paper; I haven't tried Adam  everything I've tried fails
    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

    # Build an op to do the weight clipping
    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        # if '.b' not in var.name:
        # if True:
        if ('.BN' in var.name) or ('.Out' in var.name) or ('.1' in var.name) or ('.g' in var.name):
            print "Clipping {}".format(var.name)
            clip_bounds = [-.1, .1]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
    clip_disc_weights = tf.group(*clip_ops)

# For generating plots
frame_i = [0]
fake_data_1000 = Generator(1000)
def generate_image(true_dist):
    N_POINTS = 128

    if DATASET == 'swissroll':
        RANGE = 30
    elif DATASET == '25gaussians':
        RANGE = 8

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    points = points.reshape((-1,2))
    samples, disc_map = session.run([fake_data_1000, (disc_real)], feed_dict={real_data:points})

    plt.clf()
    # plt.imshow(disc_map.reshape((N_POINTS, N_POINTS)).T[::-1, :], extent=[-RANGE, RANGE, -RANGE, RANGE], cmap='seismic', vmin=np.min(disc_map), vmax=np.max(disc_map))
    # plt.colorbar()

    x,y = np.linspace(-RANGE, RANGE, N_POINTS), np.linspace(-RANGE, RANGE, N_POINTS)
    # print disc_map.shape
    # print len(x)
    # print len(y)
    plt.contour(x,y,disc_map.reshape((len(x), len(y))))

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+')
    plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+')
    plt.savefig('frame'+str(frame_i[0])+'.jpg')
    frame_i[0] += 1

# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':
    
        dataset = []
        for i in xrange(100000/25):
            for x in xrange(-2, 3):
                for y in xrange(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        while True:
            for i in xrange(len(dataset)/BATCH_SIZE):
                yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    elif DATASET == 'swissroll':
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
            disc_iters = 32 # you should be able to reduce this dramatically but let's be safe
            for i in xrange(disc_iters):
                _disc_cost, __disc_gradients, _ = session.run([disc_cost, disc_gradients, disc_train_op], feed_dict={real_data: _data})
                if (iteration % 100 == 0) and (i == 0):
                    _debugprints = session.run([tf.nn.moments(v, range(v.get_shape().ndims)) for n,v in debugprints], feed_dict={real_data:_data})
                    for (n,sym),v in zip(debugprints, _debugprints):
                        print "{}\tmean:{}\tstd:{}".format(n,v[0],np.sqrt(v[1]))
                if not GRADIENT_LOSS:
                    _ = session.run([clip_disc_weights])
                disc_costs.append(_disc_cost)
                all_disc_gradients.append(__disc_gradients)
        else:
            if gen_train_op is not None:
                _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_data: _data})
                gen_costs.append(_gen_cost)

        if iteration % 100 == 0:
            print "iter:{}\tdisc:{:.6f} disc_gradients:{:.3f}\tgen:{:.3f}".format(iteration, np.mean(disc_costs), np.mean(all_disc_gradients), np.mean(gen_costs))
            disc_costs, all_disc_gradients, gen_costs = [], [], []

            generate_image(_data)