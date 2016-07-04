import theano.tensor as T

# Using T.nnet.relu sometimes gives me NaNs. No idea why.
def relu(x):
    return T.switch(x < 0., 0., x)