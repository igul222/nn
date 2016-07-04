import lib
import lib.ops.linear
import lib.ops.batchnorm
import theano.tensor as T

def _ReLULayer(name, input_dim, output_dim, inputs, batchnorm=False):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        input_dim=input_dim,
        output_dim=output_dim,
        inputs=inputs,
        initialization='glorot_he',
        biases=(not batchnorm)
    )

    if batchnorm:
        output = lib.ops.batchnorm.Batchnorm(
            name+'.BN',
            input_dim=output_dim,
            inputs=output
        )

    output = T.nnet.relu(output)

    return output

def MLP(name, input_dim, hidden_dim, output_dim, n_layers, inputs):
    if n_layers < 3:
        raise Exception("An MLP with <3 layers isn't an MLP!")

    output = _ReLULayer(
        name+'.Input',
        input_dim=input_dim,
        output_dim=hidden_dim,
        inputs=inputs
    )

    for i in xrange(1,n_layers-2):
        output = _ReLULayer(
            name+'.Hidden'+str(i),
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            inputs=output
        )

    return lib.ops.linear.Linear(
        name+'.Output', 
        hidden_dim,
        output_dim, 
        output,
        initialization='glorot'
    )