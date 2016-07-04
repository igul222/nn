import numpy as np
import theano
import theano.tensor as T
import theano.gof
import theano.gpuarray
import theano.gpuarray.basic_ops

import time
import uuid

def search(node, critereon):
    """
    Traverse the Theano graph starting at `node` and return a list of all nodes
    which match the `critereon` function.
    """
    def _search(node, critereon, visited):
        if node in visited:
            return []
        visited.add(node)

        results = []
        if isinstance(node, T.Apply):
            for inp in node.inputs:
                results += _search(inp, critereon, visited)
        else: # Variable node
            if critereon(node):
                results.append(node)
            if node.owner is not None:
                results += _search(node.owner, critereon, visited)
        return results

    return _search(node, critereon, set())

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

class ContextGradsOp(theano.gof.Op):
    
    def __init__(self, grads_fn, target_ctx, n_params):
        super(ContextGradsOp, self).__init__()
        self._grads_fn = grads_fn
        self._target_ctx = target_ctx
        self._n_params = n_params

    def make_node(self, *inputs):
        inputs = [
            theano.gpuarray.as_gpuarray_variable(inp, self._target_ctx)
            for inp in inputs
        ]
        return theano.gof.Apply(
            self, 
            inputs, 
            [i.type() for i in inputs[:self._n_params]]
        )

    def perform(self, node, inputs, output_storage):
        t0 = time.time()
        grads = self._grads_fn(*inputs)
        print time.time() - t0
        for cell, grad in zip(output_storage, grads):
            cell[0] = grad

def multi(grads, params, other_contexts):
    inputs = theano.gof.graph.inputs(grads)
    inputs = [
        inp for inp in inputs
        if not (isinstance(inp, T.Constant) or inp in params)
    ]

    symbolic_params = [_to_symbolic_var(p) for p in params]

    all_context_grads = []

    for ctx_i, context in enumerate(other_contexts):

        sharded_inputs = [theano.gpuarray.basic_ops.gpu_contiguous(inp[ctx_i::len(other_contexts)+1]) for inp in inputs]

        xfer_inputs = [
            theano.gpuarray.as_gpuarray_variable(
                inp,
                context
            ) 
            for inp in sharded_inputs
        ]

        xfer_params = [
            theano.gpuarray.as_gpuarray_variable(sp, context)
            for sp in symbolic_params
        ]

        replacements = {
            x: xfer_x
            for x, xfer_x in zip(params+inputs, xfer_params+xfer_inputs)
        }
    
        # For whatever reason, theano.clone likes to make its own copies of the
        # replacement nodes we give it, so we need to dig into its generated
        # graph to grab the copies it made.

        for var in (xfer_params+xfer_inputs):
            var.name = str(uuid.uuid4())

        context_grad_graphs = [
            theano.clone(g, replace=replacements)
            for g in grads
        ]

        new_inputs = []
        for var in (xfer_params+xfer_inputs):
            for g in context_grad_graphs:
                matches = search(g, lambda x: x.name==var.name)
                if len(matches):
                    new_inputs.append(matches[0])
                    break

        if len(new_inputs) != len(xfer_params+xfer_inputs):
            raise Exception()

        grads_fn = theano.function(
            new_inputs,
            [
                theano.Out(
                    g.transfer(context),
                    borrow=True
                )
                for g in context_grad_graphs
            ]
        )

        context_grads_op = ContextGradsOp(grads_fn, context, len(params))
        context_grads = context_grads_op(*(params+sharded_inputs))

        if not (isinstance(context_grads, list) or isinstance(context_grads, tuple)):
            context_grads = [context_grads]

        all_context_grads.append(context_grads)

    # context -> grad to grad -> context
    all_context_grads = zip(*all_context_grads)

    for i in xrange(len(all_context_grads)):
        all_context_grads[i] = [g.transfer(None) for g in all_context_grads[i]]

    # # Also schedule work on the main GPU
    # for i in xrange(len(all_context_grads)):
    #     sharded_inputs = [
    #         inp[len(other_contexts)::len(other_contexts)+1]
    #         for inp in inputs
    #     ]
    #     all_context_grads[i].append(
    #         theano.clone(
    #             grads[i],
    #             replace={inp:si for inp,si in zip(inputs,sharded_inputs)}
    #         )
    #     )

    avg_grads = [
        reduce(lambda a,b: a+b, gs) / float(len(gs))
        for gs in all_context_grads
    ]

    return avg_grads