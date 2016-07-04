import numpy as np
import theano.tensor as T

def gaussian_nll(x, mu, log_sigma):
    sigma = T.exp(log_sigma)
    return (
        lib.floatX(0.5*np.log(2*np.pi)) + 
        (2*log_sigma) + 
        ( ((x-mu)**2) / (2*(sigma**2)) )
    )