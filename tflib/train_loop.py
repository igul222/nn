import numpy as np
import tensorflow as tf

import itertools
import time
import collections

def train_loop(
    session,
    inputs,
    cost,
    train_data,
    times,
    prints=[],
    test_data=None,
    callback=None,
    optimizer=tf.train.AdamOptimizer(),
    inject_total_iters=False
    ):

    prints = [('cost', cost)] + prints

    grads_and_vars = optimizer.compute_gradients(
        cost,
        colocate_gradients_with_ops=True
    )

    capped_gvs = [
        (tf.clip_by_value(grad, -1., 1.), var)
        for grad, var in grads_and_vars
    ]

    train_op = optimizer.apply_gradients(capped_gvs)

    def train_fn(input_vals):
        return session.run(
            [p[1] for p in prints] + [train_op],
            feed_dict={sym:real for sym, real in zip(inputs, input_vals)}
        )[:-1]

    def eval_fn(input_vals):
        return session.run(
            [p[1] for p in prints],
            feed_dict={sym:real for sym, real in zip(inputs, input_vals)}
        )

    print "Initializing variables"
    session.run(tf.initialize_all_variables())

    total_iters = 0
    total_seconds = 0.
    last_print = 0
    last_gen = 0
    all_outputs = []
    all_stats = []
    run_times = []

    for epoch in itertools.count():
        generator = train_data()
        while True:
            try:
                input_vals = generator.next()
            except StopIteration:
                break

            if inject_total_iters:
                input_vals = [np.int32(total_iters)] + list(input_vals)

            start_time = time.time()
            outputs = train_fn(input_vals)
            run_time = time.time() - start_time
            total_seconds += run_time
            total_iters += 1
            run_times.append(run_time)

            all_outputs.append(outputs)

            if (times['mode']=='iters' and total_iters-last_print == times['print_every']) or \
                (times['mode']=='seconds' and total_seconds-last_print >= times['print_every']):

                mean_outputs = np.array(all_outputs).mean(axis=0)

                if test_data is not None:

                    if inject_total_iters:
                        test_outputs = [
                            eval_fn([np.int32(total_iters)] + list(input_vals)) 
                            for input_vals in test_data()
                        ]
                    else:
                        test_outputs = [
                            eval_fn(input_vals) 
                            for input_vals in test_data()
                        ]

                    test_mean_outputs = np.array(test_outputs).mean(axis=0)

                stats = collections.OrderedDict()
                stats['epoch'] = epoch
                stats['iters'] = total_iters
                for i,p in enumerate(prints):
                    stats['train '+p[0]] = mean_outputs[i]
                if test_data is not None:
                    for i,p in enumerate(prints):
                        stats['test '+p[0]] = test_mean_outputs[i]
                stats['secs'] = total_seconds
                stats['secs/iter'] = np.mean(run_times)

                print_str = ""
                for k,v in stats.items():
                    if isinstance(v, int):
                        print_str += "{}:{}\t".format(k,v)
                    else:
                        print_str += "{}:{:.4f}\t".format(k,v)
                print print_str[:-1] # omit the last \t

                all_stats.append(stats)

                all_outputs = []
                run_times = []
                last_print += times['print_every']

            if callback:
                if (times['mode']=='iters' and total_iters-last_gen==times['callback_every']) or \
                    (times['mode']=='seconds' and total_seconds-last_gen >= times['callback_every']):

                    tag = "iters{}_time{}".format(total_iters, total_seconds)
                    if callback is not None:
                        callback(tag)
                    if save_params:
                        lib.save_params('params_{}.pkl'.format(tag))

                    last_gen += times['callback_every']

            if (times['mode']=='iters' and total_iters == times['stop_after']) or \
                (times['mode']=='seconds' and total_seconds >= times['stop_after']):

                print "Done!"

                try: # This only matters on Ishaan's computer
                    import experiment_tools
                    experiment_tools.send_sms("done!")
                except ImportError:
                    pass

                return all_stats