def _to_symbolic_var(x):
    if x.ndim == 0:
        return T.scalar()
    elif x.ndim == 1:
        return T.vector()
    elif x.ndim == 2:
        return T.matrix()
    elif x.ndim == 3:
        return T.tensor3()
    elif x.ndim == 4:
        return T.tensor4()
    elif x.ndim == 5:
        return T.tensor5()
    else:
        raise Exception()

def _serialize_graph(inputs, outputs):
    pass

def _deserialize_graph(dump):
    pass