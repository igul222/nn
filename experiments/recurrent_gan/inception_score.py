import os, sys
sys.path.append(os.getcwd())

if __name__ == '__main__':
    try: # This only matters on Ishaan's computer
        import experiment_tools
        experiment_tools.wait_for_gpu(tf=True, n_gpus=1)
    except ImportError:
        pass

import tflib as lib
import tflib.train_loop
import tflib.mnist
import tflib.ops.mlp

import numpy as np
import tensorflow as tf

import functools
import os

LR = 1e-3
BATCH_SIZE = 1000

TIMES = {
    'mode': 'iters',
    'print_every': 1*50,
    'stop_after': 50*50,
    'test_every': 1*50
}

def build_model(graph):
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, 784])
        targets = tf.placeholder(tf.int32, shape=[None])

        logits = lib.ops.mlp.MLP(
            'InceptionScore.MLP',
            input_dim=784,
            hidden_dim=512,
            output_dim=10,
            n_layers=4,
            inputs=inputs
        )

        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
        )

        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.to_int32(tf.argmax(logits, dimension=1)),
                    targets
                ),
                tf.float32
            )
        )

        softmax = tf.nn.softmax(tf.cast(logits, tf.float64))
        # From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
        kl = softmax * (tf.log(softmax) - tf.log(tf.reduce_mean(softmax, reduction_indices=[0], keep_dims=True)))
        inception_score = tf.exp(tf.reduce_mean(tf.reduce_sum(kl, reduction_indices=[1])))

    return (inputs, targets, cost, acc, inception_score)

def train_model(graph, session, model):
    inputs, targets, cost, acc, inception_score = model

    lib.print_model_settings(locals().copy())

    train_data, dev_data, test_data = lib.mnist.load(
        BATCH_SIZE,
        BATCH_SIZE
    )

    with graph.as_default():
        lib.train_loop.train_loop(
            session=session,
            inputs=[inputs, targets],
            cost=cost,
            prints=[
                ('acc', acc),
                ('inception', inception_score)
            ],
            optimizer=tf.train.AdamOptimizer(LR),
            train_data=train_data,
            test_data=dev_data,
            times=TIMES
        )

def run_model(graph, session, model, data):
    inputs, targets, cost, acc, inception_score = model
    return session.run(inception_score, feed_dict={inputs: data})

class InceptionScore(object):
    def __init__(self):
        self._graph = tf.Graph()
        self._model = build_model(self._graph)
        self._session = tf.Session(graph=self._graph)

        if os.path.isfile('/tmp/inception_score.ckpt'):
            print "Inception score: Loading saved model weights..."
            with self._graph.as_default():
                tf.train.Saver().restore(self._session, '/tmp/inception_score.ckpt')
        else:
            print "Inception score: No saved weights found, training model..."
            train_model(self._graph, self._session, self._model)
            with self._graph.as_default():
                tf.train.Saver().save(self._session, '/tmp/inception_score.ckpt')

    def score(self, data):
        return run_model(self._graph, self._session, self._model, data)

if __name__ == '__main__':
    train_data, dev_data, test_data = lib.mnist.load(
        BATCH_SIZE,
        BATCH_SIZE
    )

    test_batches = []
    for (images, targets) in test_data():
        test_batches.append(images)
    all_test_images = np.concatenate(test_batches, axis=0)

    print "Test set inception score: {}".format(
        InceptionScore().score(all_test_images)
    )