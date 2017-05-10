"""Generative Adversarial Network for MNIST."""

import os, sys
sys.path.append(os.getcwd())

N_GPUS = 1

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=N_GPUS)
except ImportError:
    pass

import tflib as lib
import tflib.debug
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.adamax
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.ops.layernorm
import tflib.cifar10
import tflib.lsun_bedrooms
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools

BATCH_SIZE = 64
ITERS = 100000
# DIM = 32
# DIM_G = 32
MODE = 'wgan-gp' # dcgan, wgan, wgan-gp
DIM_G = 128
DIM_D = 128
# These settings apply only to WGAN-GP
EXTRA_DEPTH_G = 0
EXTRA_DEPTH_D = 0
NORMALIZATION_G = True
NORMALIZATION_D = False
# DIM = 64
OUTPUT_DIM = 3072
MOMENTUM_G = 0.
MOMENTUM_D = 0.
LR = 2e-4
DECAY = True
ARCHITECTURE = 'resnet_fixed_filters'

GAUSSIAN_DROPOUT = False
GAUSSIAN_DROPOUT_STD = 0.5
VANILLA_DROPOUT = False
VANILLA_DROPOUT_P_KEEP = 0.7

RHO = 0
GEN_BS_MULTIPLE = 1
NO_RESIDUALS = False
BN_AFTER_RELU = False
MEAN_POOL = True
UPSAMPLE_CONV = True
GLOBAL_MEAN_POOL = False

CONDITIONAL = False

if CONDITIONAL and (not NORMALIZATION_D):
    print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

def GeneratorAndDiscriminator():
    if ARCHITECTURE == 'resnet_fixed_filters':
        return FFResnetGenerator2, FFResnetDiscriminator2
    elif ARCHITECTURE == 'resnet':
        return ResnetGenerator, ResnetDiscriminator
    # return DCGANGenerator, DCGANDiscriminator

lib.print_model_settings(locals().copy())

def VanillaDropout(x):
    x_shape = tf.shape(x)
    noise_shape = tf.stack([x_shape[0], x_shape[1], 1, 1])
    return tf.nn.dropout(x, VANILLA_DROPOUT_P_KEEP, noise_shape)

def GaussianDropout(x):
    x_shape = tf.shape(x)
    noise = tf.random_normal([x_shape[0], x_shape[1], 1, 1])
    return x*(1+(GAUSSIAN_DROPOUT_STD*noise))

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs,labels=None):
    if not CONDITIONAL:
        labels = None

    if MODE == 'wgan-gp':
        if ('Discriminator' in name) and NORMALIZATION_D:
            return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
        elif ('Generator' in name) and NORMALIZATION_G:
            if labels is not None:
                return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
            else:
                return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
        else:
            return inputs
    else:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return nonlinearity(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = lib.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        if MEAN_POOL:
            conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = ConvMeanPool
        else:
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, stride=2)
            conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
    elif resample=='up':
        if UPSAMPLE_CONV:
            conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = UpsampleConv
        else:
            conv_1        = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = SubpixelConv2D
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if not NO_RESIDUALS:
        if output_dim==input_dim and resample==None:
            shortcut = inputs # Identity skip-connection
        else:
            shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    if ('Discriminator' in name) and (not no_dropout):
        if GAUSSIAN_DROPOUT:
            output = GaussianDropout(output)
        if VANILLA_DROPOUT:
            output = VanillaDropout(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)
    if ('Discriminator' in name) and (not no_dropout):
        if GAUSSIAN_DROPOUT:
            output = GaussianDropout(output)
        if VANILLA_DROPOUT:
            output = VanillaDropout(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    if NO_RESIDUALS:
        return output
    else:
        return shortcut + (.3*output)

def DCGANGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM_G, noise)
    if NORMALIZATION_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM_G, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM_G, 2*DIM_G, 5, output)
    if NORMALIZATION_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM_G, DIM_G, 5, output)
    if NORMALIZATION_G:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM_G, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def DCGANDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM_D, 5, output, stride=2)
    output = LeakyReLU(output)

    # output = GaussianDropout(output, [-1, -1, 1, 1])
    # output = tf.nn.dropout(output, KEEP_PROB, noise_shape=[BATCH_SIZE, DIM, 1, 1])

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM_D, 2*DIM_D, 5, output, stride=2)
    Normalize('Discriminator.BN2', output)
    output = LeakyReLU(output)

    # output = GaussianDropout(output, [-1, -1, 1, 1])
    # output = tf.nn.dropout(output, KEEP_PROB, noise_shape=[BATCH_SIZE, 2*DIM, 1, 1])

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM_D, 4*DIM_D, 5, output, stride=2)
    output = Normalize('Discriminator.BN3', output)
    output = LeakyReLU(output)

    # output = GaussianDropout(output, [-1, -1, 1, 1])
    # output = tf.nn.dropout(output, KEEP_PROB, noise_shape=[BATCH_SIZE, 4*DIM, 1, 1])

    output = tf.reshape(output, [-1, 4*4*4*DIM_D])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM_D, 1, output)

    return tf.reshape(output, [-1])

def ResnetGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, 4*DIM_G, 4, 4])

    for i in xrange(EXTRA_DEPTH_G):
        output = ResidualBlock('Generator.1X_{}'.format(i), 4*DIM_G, 4*DIM_G, 3, output, resample=None)
    output = ResidualBlock('Generator.1', 4*DIM_G, 2*DIM_G, 3, output, resample='up')
    for i in xrange(EXTRA_DEPTH_G):
        output = ResidualBlock('Generator.2X_{}'.format(i), 2*DIM_G, 2*DIM_G, 3, output, resample=None)
    output = ResidualBlock('Generator.2', 2*DIM_G, DIM_G, 3, output, resample='up')
    for i in xrange(EXTRA_DEPTH_G):
        output = ResidualBlock('Generator.3X_{}'.format(i), DIM_G, DIM_G, 3, output, resample=None)

    output = ResidualBlock('Generator.3', DIM_G, DIM_G/2, 3, output, resample='up', no_dropout=True)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G/2, 3, 1, output, he_init=False)

    # output = Normalize('Generator.OutputN', output)
    # output = nonlinearity(output)
    # output = lib.ops.deconv2d.Deconv2D('Generator.Output', DIM_G, 3, 5, output, he_init=False)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def ResnetDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM_D/2, 1, output, he_init=False)
    output = ResidualBlock('Discriminator.1', DIM_D/2, DIM_D, 3, output, resample='down', no_dropout=True)

    # output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM_D, 5, output, he_init=False, stride=2)

    for i in xrange(EXTRA_DEPTH_D):
        output = ResidualBlock('Discriminator.1X_{}'.format(i), DIM_D, DIM_D, 3, output, resample=None)
    output = ResidualBlock('Discriminator.2', DIM_D, 2*DIM_D, 3, output, resample='down')
    for i in xrange(EXTRA_DEPTH_D):
        output = ResidualBlock('Discriminator.2X_{}'.format(i), 2*DIM_D, 2*DIM_D, 3, output, resample=None)
    output = ResidualBlock('Discriminator.3', 2*DIM_D, 4*DIM_D, 3, output, resample='down')
    for i in xrange(EXTRA_DEPTH_D):
        output = ResidualBlock('Discriminator.3X_{}'.format(i), 4*DIM_D, 4*DIM_D, 3, output, resample=None)

    output = Normalize('Discriminator.OutputN', output)
    output = nonlinearity(output)
    output = tf.reshape(output, [-1, 4*4*4*DIM_D])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM_D, 1, output)

    return tf.reshape(output, [-1])

def FFResnetGenerator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])

    for i in xrange(EXTRA_DEPTH_G):
        output = ResidualBlock('Generator.1X_{}'.format(i), DIM_G, DIM_G, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    for i in xrange(EXTRA_DEPTH_G):
        output = ResidualBlock('Generator.2X_{}'.format(i), DIM_G, DIM_G, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    for i in xrange(EXTRA_DEPTH_G):
        output = ResidualBlock('Generator.3X_{}'.format(i), DIM_G, DIM_G, 3, output, resample=None, labels=labels)

    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', no_dropout=True, labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 1, output, he_init=False)

    # output = Normalize('Generator.OutputN', output)
    # output = nonlinearity(output)
    # output = lib.ops.deconv2d.Deconv2D('Generator.Output', DIM_G, 3, 5, output, he_init=False)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def FFResnetDiscriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM_D, 1, output, he_init=False)
    output = ResidualBlock('Discriminator.1', DIM_D, DIM_D, 3, output, resample='down', labels=labels)#, no_dropout=True)

    # output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM_D, 5, output, he_init=False, stride=2)

    for i in xrange(EXTRA_DEPTH_D):
        output = ResidualBlock('Discriminator.1X_{}'.format(i), DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    for i in xrange(EXTRA_DEPTH_D):
        output = ResidualBlock('Discriminator.2X_{}'.format(i), DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    for i in xrange(EXTRA_DEPTH_D):
        output = ResidualBlock('Discriminator.3X_{}'.format(i), DIM_D, DIM_D, 3, output, resample=None, labels=labels)

    if GLOBAL_MEAN_POOL:
        output = ResidualBlock('Discriminator.PreGMPResblock', DIM_D, DIM_D, 3, output, resample=None, labels=labels) # Extra resblock
        output = Normalize('Discriminator.OutputN', output)
        output = nonlinearity(output)
        output = tf.reduce_mean(output, axis=[2,3])
        output = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    else:
        output = Normalize('Discriminator.OutputN', output)
        output = nonlinearity(output)
        output = tf.reshape(output, [-1, 4*4*DIM_D])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*DIM_D, 1, output)

    return tf.reshape(output, [-1])

def FFResnetGenerator2(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])

    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)

    output = Normalize('Generator.OutputN', output)

    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 1, output, he_init=False)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def FFResnetDiscriminator2(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM_D, 1, output, he_init=False)
    output = ResidualBlock('Discriminator.1', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = Normalize('Discriminator.OutputN', output)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)

    return tf.reshape(output, [-1])


with tf.Session() as session:

    Generator, Discriminator = GeneratorAndDiscriminator()

    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    if (len(DEVICES)%2==0) and (len(DEVICES)>=2) and (MODE == 'wgan-gp'):

        labels_splits = lib.split(all_real_labels, len(DEVICES), axis=0)

        fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                fake_data_splits.append(Generator(BATCH_SIZE/len(DEVICES), labels_splits[i]))

        # TODO finish implementing from here onward
        all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])
        all_real_data_splits = lib.split(all_real_data, len(DEVICES), axis=0)

        DEVICES_B = DEVICES[:len(DEVICES)/2]
        DEVICES_A = DEVICES[len(DEVICES)/2:]

        disc_costs = []
        for i, device in enumerate(DEVICES_A):
            with tf.device(device):
                real_and_fake_data = lib.concat([
                    all_real_data_splits[i], 
                    all_real_data_splits[len(DEVICES_A)+i], 
                    fake_data_splits[i], 
                    fake_data_splits[len(DEVICES_A)+i]
                ], axis=0)
                real_and_fake_labels = lib.concat([
                    labels_splits[i], 
                    labels_splits[len(DEVICES_A)+i],
                    labels_splits[i],
                    labels_splits[len(DEVICES_A)+i]
                ], axis=0)
                disc_all = Discriminator(real_and_fake_data, real_and_fake_labels)
                disc_real = disc_all[:BATCH_SIZE/len(DEVICES_A)]
                disc_fake = disc_all[BATCH_SIZE/len(DEVICES_A):]
                disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))

        for i, device in enumerate(DEVICES_B):
            with tf.device(device):
                real_data = lib.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
                fake_data = lib.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
                labels = lib.concat([
                    labels_splits[i], 
                    labels_splits[len(DEVICES_A)+i],
                ], axis=0)
                alpha = tf.random_uniform(
                    shape=[BATCH_SIZE/len(DEVICES_A),1], 
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha*differences)
                gradients = tf.gradients(Discriminator(interpolates, labels), [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
                disc_costs.append(gradient_penalty)

        disc_cost = tf.add_n(disc_costs) / len(DEVICES_A)

        disc_params = lib.params_with_name('Discriminator.')
        disc_cost += RHO * tf.add_n([tf.nn.l2_loss(x) for x in disc_params if not x.name.endswith('.b')])

        if DECAY:
            decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
        else:
            decay = 1.
        disc_train_op = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=MOMENTUM_D, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        gen_costs = []
        for device in DEVICES:
            with tf.device(device):
                n_samples = GEN_BS_MULTIPLE * BATCH_SIZE / len(DEVICES)
                fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
                gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)))
        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        gen_train_op = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=MOMENTUM_G, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)

        clip_disc_weights = None

    else:

        real_data = 2*((tf.cast(all_real_data_int, tf.float32)/255.)-.5)
        fake_data = Generator(BATCH_SIZE, all_real_labels)

        if MODE == 'wgan':
            disc_real = Discriminator(real_data, all_real_labels)
            disc_fake = Discriminator(fake_data, all_real_labels)

            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                print "Clipping {}".format(var.name)
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)

            wasserstein = tf.constant(0.)
            lipschitz_penalty = tf.constant(0.)

        elif MODE == 'wgan-gp':
            disc_all = Discriminator(lib.concat([real_data, fake_data],0), lib.concat([all_real_labels,all_real_labels],axis=0))
            disc_real = disc_all[:tf.shape(real_data)[0]]
            disc_fake = disc_all[tf.shape(real_data)[0]:]
            # disc_real = Discriminator(real_data)
            # disc_fake = Discriminator(fake_data)

            fake_labels = tf.cast(tf.random_uniform([BATCH_SIZE*GEN_BS_MULTIPLE])*10, tf.int32)
            gen_cost = -tf.reduce_mean(Discriminator(Generator(BATCH_SIZE*GEN_BS_MULTIPLE,fake_labels), fake_labels))
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE,1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates, all_real_labels), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            lipschitz_penalty = tf.reduce_mean((slopes-1.)**2)
            wasserstein = disc_cost
            disc_cost += 10*lipschitz_penalty

            disc_params = lib.params_with_name('Discriminator.')
            disc_cost += RHO * tf.add_n([tf.nn.l2_loss(x) for x in disc_params if not x.name.endswith('.b')])

            if DECAY:
                decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
            else:
                decay = 1.

            gen_train_op = tf.train.AdamOptimizer(learning_rate=LR, beta1=MOMENTUM_G, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
            disc_train_op = tf.train.AdamOptimizer(learning_rate=LR, beta1=MOMENTUM_D, beta2=0.9).minimize(disc_cost, var_list=disc_params)

            clip_disc_weights = None

        elif MODE == 'dcgan':
            disc_real = Discriminator(real_data, all_real_labels)
            disc_fake = Discriminator(fake_data, all_real_labels)

            gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
            disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
            disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
            disc_cost /= 2.

            gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
            disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))

    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), 'samples_{}.png'.format(frame))

    fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
    samples_100 = Generator(100, fake_labels_100)
    def get_inception_score(n):
        all_samples = []
        for i in xrange(n/100):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
        return lib.inception_score.get_inception_score(list(all_samples))

    train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE)
    def inf_train_gen():
        while True:
            for images,_labels in train_gen():
                yield images,_labels


    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run([gen_train_op], feed_dict={_iteration:iteration})

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = 5

        for i in xrange(disc_iters):
            _data,_labels = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('cost', _disc_cost)

        if iteration % 10 == 9:
            lib.plot.plot('time', time.time() - start_time)

        # # Calculate inception score every 100 iters
        if iteration % 100 == 99:
            inception_score = get_inception_score(200)
            lib.plot.plot('inception', inception_score[0])

        if iteration % 10000 == 9999:
            inception_score = get_inception_score(50000)
            lib.plot.plot('inception_50k', inception_score[0])
            lib.plot.plot('inception_50k_std', inception_score[1])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_labels in dev_gen():
                _dev_disc_cost = session.run([disc_cost], feed_dict={all_real_data_int: images,all_real_labels:_labels})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        if (iteration < 500) or (iteration % 1000 == 999):
            lib.plot.flush()

        lib.plot.tick()