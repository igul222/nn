"""
Unrolled VAE
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=1)
except ImportError:
    pass

import tflib as lib
import tflib.debug
import tflib.ops.kl_unit_gaussian
import tflib.ops.linear

import tflib.mnist_binarized

import tensorflow as tf
import numpy as np

DIM = 512
LATENT_DIM = 32

BATCH_SIZE = 1000
TRAIN_ITERS = 100*500
# TRAIN_ITERS = 10

# MODE = 'vanilla'
# UNROLL_STEPS = 0

# Most fair comparison is unroll_decoder vs unroll_decoder with a tf.stop_gradient in the middle
MODE = 'unroll_decoder'
UNROLL_STEPS = 3

# OPTIMIZER = 'sgd'
# LEARNING_RATE = 1e-2

OPTIMIZER = 'adam'
LEARNING_RATE = 1e-4

train_data, dev_data, test_data = lib.mnist_binarized.load(BATCH_SIZE, BATCH_SIZE)

lib.print_model_settings(locals().copy())

lib.ops.linear.enable_default_weightnorm()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    images = tf.placeholder(tf.float32, shape=[None, 1, 28, 28], name='images')
    flat_images = tf.reshape(images, [-1, 784])

    # For some reason, ReLU immediately gives nans in certain configurations.
    # LeakyReLU and ELU seem to work fine.
    def nonlinearity(x):
        return tf.nn.elu(x)

    def Encoder(inputs):
        output = lib.ops.linear.Linear('Encoder.1', 784, DIM, inputs)
        output = nonlinearity(output)
        output = lib.ops.linear.Linear('Encoder.2', DIM, DIM, output)
        output = nonlinearity(output)
        # lib.debug.print_stats('Encoder.3_Before', output)
        output = lib.ops.linear.Linear('Encoder.3', DIM, 2*LATENT_DIM, output)
        # lib.debug.print_stats('Encoder.3_After', output)
        return output

    def Decoder(latents):
        # lib.debug.print_stats('Decoder.1_Before', latents)        
        output = lib.ops.linear.Linear('Decoder.1', LATENT_DIM, DIM, latents)
        # lib.debug.print_stats('Decoder.1_After', output)
        output = nonlinearity(output)
        # lib.debug.print_stats('Decoder.1_After2', output)
        output = lib.ops.linear.Linear('Decoder.2', DIM, DIM, output)
        output = nonlinearity(output)
        output = lib.ops.linear.Linear('Decoder.3', DIM, 784, output)
        return output

    def build_forward_graph(flat_images_split):
        # flat_images_split = flat_images
        mu, log_sigma = tf.split(1, 2, Encoder(flat_images_split))
        sigma = tf.nn.softplus(log_sigma)
        log_sigma = tf.log(sigma)
        # sigma = tf.exp(log_sigma)

        # lib.debug.print_stats('mu', mu)
        # lib.debug.print_stats('sigma', sigma)
        # lib.debug.print_stats('log_sigma', log_sigma)

        eps = tf.random_normal(tf.shape(mu))
        latents = mu + (eps * sigma)

        logits = Decoder(latents)

        kl_cost = tf.reduce_mean(
            lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                mu, 
                log_sigma,
                sigma
            )
        ) * float(LATENT_DIM)

        reconst_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits, flat_images_split)
        ) * float(784)

        cost = reconst_cost + kl_cost
        return cost, kl_cost

    cost, kl_cost = build_forward_graph(flat_images[0::UNROLL_STEPS+1])

    def make_adam_vars(name, params):
        t = lib.param(name+'.t', np.float32(0.))
        ms, vs = [], []
        for i, param in enumerate(params):
            ms.append(lib.param(name+'.m_{}'.format(i), np.zeros(param.get_shape(), dtype='float32')))
            vs.append(lib.param(name+'.v_{}'.format(i), np.zeros(param.get_shape(), dtype='float32')))
        return (t, ms, vs)

    def apply_adam(params, grads, adam_vars):
        t, ms, vs = adam_vars

        beta1=0.9
        beta2=0.999
        epsilon=1e-8

        t = t + 1.
        a_t = LEARNING_RATE*tf.sqrt(1.-beta2**t)/(1.-beta1**t)

        new_ms = []
        new_vs = []
        new_params = {}

        for param, g_t, m_prev, v_prev in zip(params, grads, ms, vs):
            m_t = beta1*m_prev + (1.-beta1)*g_t

            if OPTIMIZER == 'adam':
                v_t = beta2*v_prev + (1.-beta2)*g_t**2
                step = a_t*m_t/(tf.sqrt(v_t) + epsilon)
            elif OPTIMIZER == 'adamax':
                v_t = tf.maximum(beta2*v_prev + epsilon, tf.abs(g_t))
                step = LEARNING_RATE * m_t / v_t
            elif OPTIMIZER == 'sgd':
                v_t = v_prev
                step = LEARNING_RATE * g_t
            else:
                raise Exception()

            new_ms.append(m_t)
            new_vs.append(v_t)
            new_params[param] = param - step

        return new_params, (t, new_ms, new_vs)

    def make_adam_var_update_ops(old, new):
        old_t, old_ms, old_vs = old
        new_t, new_ms, new_vs = new
        updates = []
        updates.append(tf.assign(old_t, new_t))
        for old, new in zip(old_ms + old_vs, new_ms + new_vs):
            updates.append(tf.assign(old, new))
        return updates

    def clip_grads(grads):
        return tf.clip_by_global_norm(grads, 5.0)[0]

    enc_params = lib.params_with_name('Encoder')
    dec_params = lib.params_with_name('Decoder')

    enc_adam_vars = make_adam_vars('Encoder_Adam', enc_params)
    dec_adam_vars = make_adam_vars('Decoder_Adam', dec_params)

    if MODE == 'vanilla':
        enc_grads = tf.gradients(cost, enc_params)
        dec_grads = tf.gradients(cost, dec_params)

        enc_updates, new_enc_adam_vars = apply_adam(enc_params, enc_grads, enc_adam_vars)
        dec_updates, new_dec_adam_vars = apply_adam(dec_params, dec_grads, dec_adam_vars)

        update_ops = []
        for old, new in enc_updates.items() + dec_updates.items():
            update_ops.append(tf.assign(old, new))

        update_ops.extend(make_adam_var_update_ops(enc_adam_vars, new_enc_adam_vars))
        update_ops.extend(make_adam_var_update_ops(dec_adam_vars, new_dec_adam_vars))

    elif MODE == 'unroll_decoder':

        dec_grads = clip_grads(tf.gradients(cost, dec_params))
        dec_updates, new_dec_adam_vars = apply_adam(dec_params, dec_grads, dec_adam_vars)

        all_costs = [cost]

        new_cost = cost
        new_dec_params = [dec_updates[p] for p in dec_params]
        new_dec_updates = dec_updates
        newer_dec_adam_vars = new_dec_adam_vars
        for i in xrange(UNROLL_STEPS - 1):
            lib.alias_params(new_dec_updates)
            new_cost, new_kl_cost = build_forward_graph(flat_images[i+1::UNROLL_STEPS])
            all_costs.append(new_cost)
            new_dec_updates, newer_dec_adam_vars = apply_adam(
                new_dec_params, 
                clip_grads(tf.gradients(new_cost, new_dec_params)),
                newer_dec_adam_vars
            )
            new_dec_params = [new_dec_updates[p] for p in new_dec_params]

        lib.alias_params(new_dec_updates)
        new_cost, new_kl_cost = build_forward_graph(flat_images[UNROLL_STEPS::UNROLL_STEPS+1])

        enc_grads = clip_grads(tf.gradients(tf.reduce_mean(tf.pack(all_costs)), enc_params))
        enc_updates, new_enc_adam_vars = apply_adam(enc_params, enc_grads, enc_adam_vars)

        update_ops = []
        for old, new in enc_updates.items() + ({old:new for old,new in zip(dec_params, new_dec_params)}).items():
            update_ops.append(tf.assign(old, new))

        update_ops.extend(make_adam_var_update_ops(enc_adam_vars, new_enc_adam_vars))
        update_ops.extend(make_adam_var_update_ops(dec_adam_vars, newer_dec_adam_vars))

    else:
        raise Exception()

    session.run(tf.initialize_all_variables())

    def inf_train_gen():
        while True:
            for data in train_data():
                yield data


    train_iters = 0
    costs, kl_costs = [], []
    for (_images,) in inf_train_gen():
        lib.debug.print_all_stats({images:_images})
        _outputs = session.run([cost, kl_cost] + update_ops, feed_dict={images: _images})
        costs.append(_outputs[0])
        kl_costs.append(_outputs[1])
        # if True:
        if train_iters % 500 == 499:
            print "iter {}\tcost {}\tkl {}".format(train_iters, np.mean(costs), np.mean(kl_costs))
            costs, kl_costs = [], []
        train_iters += 1
        if train_iters == TRAIN_ITERS:
            break