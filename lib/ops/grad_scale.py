import theano

class GradScale(theano.compile.ViewOp):
    __props__ = ()
    def __init__(self, scale_amount):
        # We do not put those member in __eq__ or __hash__
        # as they do not influence the perform of this op.
        self.scale_amount = scale_amount

    def grad(self, args, g_outs):
        return [g_out * self.scale_amount for g_out in g_outs]

def grad_scale(x, alpha):
    # return theano.gradient.disconnected_grad(x)
    return GradScale(alpha)(x)