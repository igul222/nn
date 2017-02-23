"""Generative Adversarial Network for MNIST."""

# Ideas:
# Penalize information retention by the discriminator (to prevent time-wise sparse gradients); i.e. encourage forget-gate activation
# EBGAN

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, skip=[1])
except ImportError:
    pass

import tflib as lib
import tflib.debug
import tflib.ops.linear
import tflib.ops.rnn
import tflib.ops.gru
import tflib.ops.conv1d
import tflib.ops.batchnorm

import data_tools
import tflib.save_images

import numpy as np
import tensorflow as tf
import scipy.misc
from scipy.misc import imsave

import time
import functools

BATCH_SIZE = 100
ITERS = 100000
SEQ_LEN = 20

lib.print_model_settings(locals().copy())

ALL_SETTINGS = {
    'minibatch_discrim': [True, False], # False works best
    'generator_output_mode': ['softmax', 'st_argmax', 'softmax_st_argmax', 'st_sampler', 'gumbel_sampler'],     # gumbel_sampler works best, followde by softmax_st_argmax
    'ngram_discrim': [None, 1, 2, 4], # can't tell the difference between [1], [2], [2,4]. None is bad.
    'dim': [128, 256, 512],
    'noise': ['normal', 'uniform'], # can't tell the difference. normal might have a slight edge?
    'input_noise_std': 0.0, # seems to hurt; 0.1 might help; TODO try annealing down over time
    'ngram_input_noise_std': 0.0, # also seems to hurt; TODO try annealing down over time
    'one_sided_label_smoothing': False # seems to hurt
}

# SETTINGS = experiment_tools.pick_settings(ALL_SETTINGS)
SETTINGS = {
    'minibatch_discrim': False,
    'generator_output_mode': 'softmax',
    'ngram_discrim': None,
    'dim': 512,
    'dim_g': 256,
    'noise': 'normal',
    'input_noise_std': 0.,
    'ngram_input_noise_std': 0.0,
    'one_sided_label_smoothing': False,
    'disc_lm': False,
    'extra_disc_steps': 4,
    'rnn_discrim': True,
    'simultaneous_update': False,
    'feature_matching': False, # seems to work surprisingly well, but theoretically I don't like it?
    'word_dropout': 0.,
    '2layer_generator': False,
    'wgan': True,
    'gen_lm': False # doesn't help
}

lib.print_model_settings_dict(SETTINGS)

lines, charmap, inv_charmap = data_tools.load_dataset(max_length=SEQ_LEN, max_n_examples=1000000)

# lib.ops.linear.enable_default_weightnorm()

def LeakyReLU(x, alpha=.25):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs, alpha=0.25):
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

def softmax(logits):
    softmax = tf.reshape(tf.nn.softmax(tf.reshape(logits, [-1, len(charmap)])), tf.shape(logits))
    return softmax

def st_argmax(logits):
    """straight-through argmax"""
    argmax = tf.argmax(logits, logits.get_shape().ndims-1)
    onehot = tf.one_hot(argmax, len(charmap))
    residual = onehot - logits
    onehot = logits + tf.stop_gradient(residual)
    return onehot

def softmax_st_argmax(logits):
    """softmax -> straight-through argmax"""
    return st_argmax(softmax(logits))

def st_sampler(logits):
    """straight-through stochastic sampler"""
    flat_samples = tf.reshape(tf.multinomial(tf.reshape(logits, [-1, len(charmap)]), 1), [-1])
    onehot = tf.reshape(tf.one_hot(flat_samples, len(charmap)), tf.shape(logits))

    residual = onehot - logits
    onehot = logits + tf.stop_gradient(residual)
    return onehot

def gumbel_sampler(logits):
    """gumbel-softmax -> straight-through argmax"""
    gumbel_noise = -tf.log(-tf.log(tf.random_uniform(tf.shape(logits))))
    logits += gumbel_noise
    logits /= 0.1 # gumbel temp
    gumbel_softmax = tf.reshape(tf.nn.softmax(tf.reshape(logits, [-1, len(charmap)])), tf.shape(logits))
    return st_argmax(gumbel_softmax)


def make_noise(shape):
    if SETTINGS['noise'] == 'uniform':
        return tf.random_uniform(shape=shape, minval=-np.sqrt(3), maxval=np.sqrt(3))
    elif SETTINGS['noise'] == 'normal':
        return tf.random_normal(shape)
    else:
        raise Exception()

# def SubpixelConv1D(*args, **kwargs):
#     kwargs['output_dim'] = 2*kwargs['output_dim']
#     output = lib.ops.conv1d.Conv1D(*args, **kwargs)
#     output = tf.transpose(output, [0,2,1])
#     output = tf.reshape(output, [output.get_shape()[0], -1, kwargs['output_dim']])
#     output = tf.transpose(output, [0,2,1])
#     return output

def Generator(n_samples, prev_outputs=None):
    output = make_noise(
        shape=[n_samples, 128]
    )

    output = lib.ops.linear.Linear('Generator.Input', 128, SEQ_LEN*SETTINGS['dim_g'], output)
    output = tf.reshape(output, [-1, SETTINGS['dim_g'], SEQ_LEN])

    output = ResBlockG('Generator.1', output)
    output = ResBlockG('Generator.2', output)
    output = ResBlockG('Generator.3', output)
    output = ResBlockG('Generator.4', output)
    output = ResBlockG('Generator.5', output)

    output = lib.ops.conv1d.Conv1D('Generator.Output', SETTINGS['dim_g'], len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])

    output = softmax(output)

    return output, None

# def Generator(n_samples, prev_outputs=None):
#     noise = make_noise(
#         shape=[n_samples, SEQ_LEN, 8]
#     )

#     h0_noise = make_noise(
#         shape=[n_samples, SETTINGS['dim_g']]
#     )

#     h0_noise_2 = make_noise(
#         shape=[n_samples, SETTINGS['dim_g']]
#     )

#     rnn_outputs = []
#     rnn_logits = []
#     last_state = h0_noise
#     last_state_2 = h0_noise_2
#     for i in xrange(SEQ_LEN):
#         if len(rnn_outputs) > 0:
#             if prev_outputs is not None:
#                 inputs = tf.concat(1, [noise[:,i], prev_outputs[:,i-1]])
#             else:
#                 inputs = tf.concat(1, [noise[:,i], rnn_outputs[-1]])
#         else:
#             inputs = tf.concat(1, [noise[:,i], tf.zeros([n_samples, len(charmap)])])

#         # print "WARNING FORCING INDEPENDENT OUTPUTS"
#         # inputs *= 0.

#         last_state = lib.ops.gru.GRUStep('Generator.1', 8+len(charmap), SETTINGS['dim_g'], inputs, last_state)
#         last_state_2 = lib.ops.gru.GRUStep('Generator.2', SETTINGS['dim_g'], SETTINGS['dim_g'], last_state, last_state_2)
#         if SETTINGS['2layer_generator']:
#             output = lib.ops.linear.Linear('Generator.Out', SETTINGS['dim_g'], len(charmap), last_state_2)
#         else:
#             output = lib.ops.linear.Linear('Generator.Out', SETTINGS['dim_g'], len(charmap), last_state)

#         rnn_logits.append(output)

#         if SETTINGS['generator_output_mode']=='softmax':
#             output = softmax(output)
#         elif SETTINGS['generator_output_mode']=='st_argmax':
#             output = st_argmax(output)
#         elif SETTINGS['generator_output_mode']=='softmax_st_argmax':
#             output = softmax_st_argmax(output)
#         elif SETTINGS['generator_output_mode']=='st_sampler':
#             output = st_sampler(output)
#         elif SETTINGS['generator_output_mode']=='gumbel_sampler':
#             output = gumbel_sampler(output)
#         else:
#             raise Exception()
        
#         rnn_outputs.append(output)

#     return tf.transpose(tf.pack(rnn_outputs), [1,0,2]), tf.transpose(tf.pack(rnn_logits), [1,0,2])

# def NgramDiscrim(n, inputs):
#     inputs += SETTINGS['ngram_input_noise_std']*tf.random_normal(tf.shape(inputs))
#     output = tf.reshape(inputs, [-1, n*len(charmap)])
#     output = ReLULayer('Discriminator.CharDiscrim_{}.FC'.format(n), n*len(charmap), SETTINGS['dim'], output)
#     # output = MinibatchLayer('Discriminator.CharDiscrim.Minibatch', SETTINGS['dim'], 32, 16, output)
#     # output = ReLULayer('Discriminator.CharDiscrim.FC2', SETTINGS['dim']+32, SETTINGS['dim'], output)
#     output = lib.ops.linear.Linear('Discriminator.CharDiscrim_{}.Output'.format(n), SETTINGS['dim'], 1, output)
#     return output

def NgramDiscrim(n, inputs):
    inputs += SETTINGS['ngram_input_noise_std']*tf.random_normal(tf.shape(inputs))
    output = tf.reshape(inputs, [-1, n, len(charmap)])
    output = lib.ops.gru.GRU('Discriminator.GRU', len(charmap), SETTINGS['dim'], output)
    features = output
    output = output[:, -1, :] # last hidden state
    output = lib.ops.linear.Linear('Discriminator.Output', SETTINGS['dim'], 1, output)
    return output, features

def GatedResBlock(name, inputs):
    output = inputs
    dim = SETTINGS['dim']
    output = LeakyReLU(output)
    output_a = lib.ops.conv1d.Conv1D(name+'.A', dim, dim, 5, output)
    output_b = lib.ops.conv1d.Conv1D(name+'.B', dim, dim, 5, output)
    output = tf.nn.sigmoid(output_a) * tf.tanh(output_b)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim, dim, 5, output)
    return inputs + output

def ResBlock(name, inputs):
    output = inputs
    dim = SETTINGS['dim']
    output = tf.nn.relu(output)
    output_a = lib.ops.conv1d.Conv1D(name+'.1', dim, dim, 3, output)
    output = tf.nn.relu(output_a)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim, dim, 3, output)
    # output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2], output)
    return inputs + (0.3*output)

def ResBlockG(name, inputs):
    output = inputs
    dim = SETTINGS['dim_g']
    output = tf.nn.relu(output)
    output_a = lib.ops.conv1d.Conv1D(name+'.1', dim, dim, 3, output)
    output = tf.nn.relu(output_a)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim, dim, 3, output)
    # output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2], output)
    return inputs + (0.3*output)

def ResBlockUpsample(name, dim_in, dim_out, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output_a = SubpixelConv1D(name+'.1', dim_in, dim_out, 3, output)
    output = tf.nn.relu(output_a)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim_out, dim_out, 3, output)
    # output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2], output)
    return output + SubpixelConv1D(name+'.skip', dim_in, dim_out, 1, inputs)

def Discriminator(inputs):
    inputs += SETTINGS['input_noise_std']*tf.random_normal(tf.shape(inputs))

    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.1', len(charmap), SETTINGS['dim'], 1, output)

    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = ResBlock('Discriminator.6', output)

    output = tf.reduce_mean(output, reduction_indices=[2])

    # output = MinibatchLayer('Discriminator.Minibatch', SETTINGS['dim'], 32, 16, output)
    # output = ReLULayer('Discriminator.FC', SETTINGS['dim']+32, SETTINGS['dim'], output)

    output = lib.ops.linear.Linear('Discriminator.Output', SETTINGS['dim'], 1, output)
    return output, None, None

# def Discriminator(inputs):
#     inputs += SETTINGS['input_noise_std']*tf.random_normal(tf.shape(inputs))
#     if SETTINGS['word_dropout'] > 0.001:
#         inputs = tf.nn.dropout(inputs, keep_prob=SETTINGS['word_dropout'], noise_shape=[BATCH_SIZE, SEQ_LEN, 1])
#     output = inputs
#     output = lib.ops.gru.GRU('Discriminator.GRU', len(charmap), SETTINGS['dim'], output)

#     features = [output]

#     # Auxiliary language model
#     language_model_output = lib.ops.linear.Linear('Discriminator.LMOutput', SETTINGS['dim'], len(charmap), output[:, :-1, :])

#     # output = tf.reduce_mean(output, reduction_indices=[1]) # global-average-pool
#     output = output[:, SEQ_LEN-1, :] # last hidden state

#     # # Auxiliary autoencoder
#     # autoencoder_hiddens = []
#     # last_state = output
#     # for i in xrange(SEQ_LEN):
#     #     inputs = tf.zeros([n_samples, 1])
#     #     last_state = lib.ops.gru.GRUStep('Discriminator.Autoencoder', 1, SETTINGS['dim'], inputs, last_state)
#     #     autoencoder_hiddens.append(last_state)
#     # autoencoder_hiddens = tf.pack(autoencoder_hiddens)

#     if SETTINGS['minibatch_discrim']:
#         output = MinibatchLayer('Discriminator.Minibatch', SETTINGS['dim'], 32, 16, output)
#         output = ReLULayer('Discriminator.FC', SETTINGS['dim']+32, SETTINGS['dim'], output)

#     output = lib.ops.linear.Linear('Discriminator.Output1', SETTINGS['dim'], SETTINGS['dim'], output)
#     output = LeakyReLU(output)
#     output = lib.ops.linear.Linear('Discriminator.Output', SETTINGS['dim'], 1, output)

#     outputs = []
#     if SETTINGS['ngram_discrim'] is not None:
#         for i in SETTINGS['ngram_discrim']:
#             ngram_output, ngram_features = NgramDiscrim(i, inputs)
#             outputs.append(ngram_output)
#             features.append(ngram_features)
#     if SETTINGS['rnn_discrim']:
#         outputs.append(output)
#     return tf.concat(0, outputs), language_model_output, features # we apply the sigmoid later

real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs, _ = Generator(BATCH_SIZE)
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

disc_real, disc_real_lm, disc_real_features = Discriminator(real_inputs) 
disc_fake, disc_fake_lm, disc_fake_features = Discriminator(fake_inputs)

# Gen objective:  push D(fake) to one
if SETTINGS['feature_matching']:
    gen_costs = [tf.reduce_mean((tf.reduce_mean(real_features, reduction_indices=[0]) - tf.reduce_mean(fake_features, reduction_indices=[0]))**2) for real_features, fake_features in zip(disc_real_features, disc_fake_features)]
    gen_cost = 0.
    for gc in gen_costs:
        gen_cost = gen_cost + gc
elif SETTINGS['wgan']:
    gen_cost = -tf.reduce_mean(disc_fake)
else:
    if SETTINGS['one_sided_label_smoothing']:
        raise Exception('check this implementation')
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
    else:
        print "warning minimax cost"
        gen_cost = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
        # gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))

# Discrim objective: push D(fake) to zero, and push D(real) to onehot
if SETTINGS['wgan']:
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    # WGAN lipschitz-penalty
    # epsilon = 1
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates)[0], [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
    # print slopes.get_shape()
    # interpolates_1 = real_data + ((alpha-epsilon)*differences)
    # interpolates_2 = real_data + ((alpha+epsilon)*differences)
    # slopes = tf.abs((Discriminator(interpolates_2)-Discriminator(interpolates_1))/(2*epsilon))
    # lipschitz_penalty = tf.reduce_mean(tf.maximum(10000000.,slopes))
    # lipschitz_penalty = tf.reduce_mean(tf.maximum(0.,(slopes-1.)**1))
    # lipschitz_penalty = tf.reduce_mean((slopes-10.))
    lipschitz_penalty = tf.reduce_mean((slopes-1.)**2)
    wgan_disc_cost = disc_cost
    disc_cost += 10*lipschitz_penalty
    lipschitz_penalty = tf.reduce_mean(slopes)


else:
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
    if SETTINGS['one_sided_label_smoothing']:
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, 0.9*tf.ones_like(disc_real)))
    else:
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
    disc_cost /= 2.

if SETTINGS['disc_lm']:
    disc_cost += tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.reshape(disc_real_lm, [-1, len(charmap)]),
            tf.reshape(real_inputs_discrete[:, 1:], [-1])
        )
    )

    # disc_cost += tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(
    #         tf.reshape(disc_fake_lm, [-1, len(charmap)]),
    #         tf.reshape(fake_inputs_discrete[:, 1:], [-1])
    #     )
    # )

if SETTINGS['gen_lm']:
    gen_cost += tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.reshape(Generator(BATCH_SIZE, real_inputs[:, :-1])[1], [-1, len(charmap)]),
            tf.reshape(real_inputs_discrete, [-1])
        )
    )

if SETTINGS['wgan']:
    # gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-4).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    # disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-4).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))
    # disc_train_op_nopenalty = tf.train.RMSPropOptimizer(learning_rate=5e-4).minimize(wgan_disc_cost, var_list=lib.params_with_name('Discriminator'))

    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

    assigns = []
    for var in lib.params_with_name('Discriminator'):
        if ('.b' not in var.name) and ('Bias' not in var.name) and ('.BN' not in var.name):
            print "Clipping {}".format(var.name)
            clip_bounds = [-.01, .01]
            assigns.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
        if '.BN.scale' in var.name:
            print "Clipping {}".format(var.name)
            clip_bounds = [0, 1]
            assigns.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))            
    clip_disc_weights = tf.group(*assigns)

else:
    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))


def iterate_dataset():
    while True:
        np.random.shuffle(lines)
        for i in xrange(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield np.array([[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]], dtype='int32')


# true_char_lm = data_tools.NgramLanguageModel(1, lines, tokenize=False)
# true_word_lm = data_tools.NgramLanguageModel(1, lines, tokenize=True)

true_char_ngram_lms = [data_tools.NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False) for i in xrange(4)]
validation_char_ngram_lms = [data_tools.NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in xrange(4)]
for i in xrange(4):
    print "validation JS for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i]))

true_char_ngram_lms = [data_tools.NgramLanguageModel(i+1, lines, tokenize=False) for i in xrange(4)]

# generator_word_lm = data_tools.NgramLanguageModel(1, lines[:3*BATCH_SIZE], tokenize=True)

# print "word precision: {}".format(true_word_lm.precision_wrt(generator_word_lm))

# tf.add_check_numerics_ops()

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in xrange(len(samples)):
            decoded = ""
            for j in xrange(len(samples[i])):
                decoded += inv_charmap[samples[i][j]]
            decoded_samples.append(decoded)
        return decoded_samples

    def callback(iteration):
        samples = []
        for i in xrange(10):
            samples.extend(generate_samples())

        jses = []
        for i in xrange(4):
            lm = data_tools.NgramLanguageModel(i+1, samples, tokenize=False)
            jses.append(lm.js_with(true_char_ngram_lms[i]))

        print "ngram JSes: {}".format(jses)
        # generator_char_lm = data_tools.NgramLanguageModel(1, samples, tokenize=False)
        # generator_word_lm = data_tools.NgramLanguageModel(1, samples, tokenize=True)
        # print "char KL: {}\tword precision: {}".format(true_char_lm.kl_to(generator_char_lm), true_word_lm.precision_wrt(generator_word_lm))

        with open('samples_{}.txt'.format(iteration), 'w') as f:
            for s in samples:
                f.write(s + "\n")

    gen = iterate_dataset()
    gen_costs = []
    disc_costs = []
    lipschitz_penalties = []

    disc_iters = SETTINGS['extra_disc_steps']+1
    fake_inputs_batch,_ = Generator(disc_iters*BATCH_SIZE)

    for iteration in xrange(ITERS):

        # TODO consolidate the separate train loops
        if SETTINGS['wgan']:

            start_time = time.time()

            _data = gen.next()

            if iteration % 2 == 0:
                _fake_inputs_batch = session.run(fake_inputs_batch)
                for i in xrange(disc_iters):
                    j = i % disc_iters
                    # if iteration < 100:
                    #     _disc_cost, _lipschitz_penalty, _ = session.run([wgan_disc_cost, lipschitz_penalty, disc_train_op_nopenalty], feed_dict={real_inputs_discrete:_data, fake_inputs: _fake_inputs_batch[BATCH_SIZE*j:BATCH_SIZE*(j+1)]})
                    # else:                        
                    _disc_cost, _lipschitz_penalty, _ = session.run([wgan_disc_cost, lipschitz_penalty, disc_train_op], feed_dict={real_inputs_discrete:_data, fake_inputs: _fake_inputs_batch[BATCH_SIZE*j:BATCH_SIZE*(j+1)]})
                    # _ = session.run([clip_disc_weights])
                    disc_costs.append(_disc_cost)
                    lipschitz_penalties.append(_lipschitz_penalty)
                    _data = gen.next()
                    # print _disc_cost
                    # if i % 100 == 99:
                    #     print np.mean(disc_costs[-100:])
            else:
                if (iteration > 100):
                    if gen_train_op is not None:
                        _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_inputs_discrete:_data})
                        gen_costs.append(_gen_cost)

            if iteration % 10 == 0:
                print "iter:{}\tdisc:{:.6f} {:.6f}\tgen:{:.6f}\ttime:{:.6f}".format(iteration, np.mean(disc_costs), np.mean(lipschitz_penalties), np.mean(gen_costs), time.time() - start_time)
                disc_costs, lipschitz_penalties, gen_costs = [], [], []

            if iteration % 100 == 0:
                print "saving"
                callback(iteration)

        else:

            start_time = time.time()
            _inputs = gen.next()
            if SETTINGS['simultaneous_update']:
                _disc_cost, _gen_cost, _, _ = session.run([disc_cost, gen_cost, disc_train_op, gen_train_op], feed_dict={real_inputs_discrete:_inputs})

                for i in xrange(SETTINGS['extra_disc_steps']):
                    _inputs = gen.next()
                    _ = session.run([disc_train_op], feed_dict={real_inputs_discrete:_inputs})

            else:
                _fake_inputs_batch = session.run(fake_inputs_batch)
                for i in xrange(disc_iters):
                    j = i % disc_iters
                    _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_inputs_discrete:_inputs, fake_inputs: _fake_inputs_batch[BATCH_SIZE*j:BATCH_SIZE*(j+1)]})
                    _inputs = gen.next()

                _gen_cost, _ = session.run([gen_cost, gen_train_op])

            end_time = time.time() - start_time

            gen_costs.append(_gen_cost)
            disc_costs.append(_disc_cost)

            if iteration % 10 == 0:
                print "iter:{}\tdisc:{:.3f},std{:.3f}\tgen:{:.3f},std{:.3f}\ttime:{:.3f}".format(
                    iteration, 
                    np.mean(disc_costs), 
                    np.std(disc_costs),
                    np.mean(gen_costs),
                    np.std(gen_costs), 
                    end_time,
                )
                gen_costs = []
                disc_costs = []

                if iteration % 100 == 0:
                    callback(iteration)