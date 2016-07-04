import lib.debug

import theano
import theano.tensor as T

def softmax_nll(x, targets):
    def logsumexp(x, axis=None, keepdims=True):
        """A numerically stable version of log(sum(exp(x)))."""
        x_max = T.max(x, axis=axis, keepdims=True)
        return x_max + T.log(
            T.sum(
                T.exp(x - x_max),
                axis=axis,
                keepdims=keepdims
            )
        )

    log_softmax = x - logsumexp(x, axis=x.ndim-1)

    # log_softmax = lib.debug.print_stats('log_softmax', log_softmax)

    log_softmax = log_softmax.reshape((-1, log_softmax.shape[-1]))
    flat_targets = targets.flatten()

    nlls = -log_softmax[T.arange(log_softmax.shape[0]), flat_targets]

    # nlls = lib.debug.print_stats('nlls', nlls)

    return nlls.reshape(targets.shape)