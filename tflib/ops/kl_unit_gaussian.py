import tensorflow as tf

def kl_unit_gaussian(mu, log_sigma):
    """
    KL divergence from a unit Gaussian prior
    based on yaost, via Alec
    """
    return -0.5 * (1 + 2 * log_sigma - mu**2 - tf.exp(2 * log_sigma))