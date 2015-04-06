import numpy
import theano

from theano import tensor
from blocks.bricks.recurrent import SimpleRecurrent, recurrent
from blocks.bricks.base import lazy, application
from blocks.algorithms import GradientDescent, RMSProp, Momentum, CompositeRule
from blocks.utils import shared_floatx_nans


class SparseFilter(SimpleRecurrent):
    @lazy
    def __init__(self, dim, input_dim, batch_size, *args, **kwargs):
        super(SparseFilter, self).__init__(dim, *args, **kwargs)
        self.dim = dim
        self.input_dim = input_dim

    def _allocate(self):
        self.params.append(shared_floatx_nans((self.dim,
            self.input_dim), name="W"))

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        if state_name == 'states':
            dim = self.get_dim('states')
            zeros = numpy.zeros((batch_size, dim))
            return theano.shared(zeros.astype(theano.config.floatX))
        return super(SparseFilter, self).initial_state(state_name,
                batch_size, *args, **kwargs)

    def get_dim(self, name):
        if name in (SparseFilter.apply.states +
                SparseFilter.apply.outputs[1:]):
            return self.dim
        elif name == 'outputs':
            return self.input_dim
        return super(SparseFilter, self).get_dim(name)

    @recurrent(sequences=[], states=['states', 'accum_1', 'accum_2'],
               outputs=['outputs', 'states',
                  'accum_1', 'accum_2'],
               contexts=['inputs'])
    def apply(self, inputs=None, states=None, accum_1=None, accum_2=None):
        """ The outputs of this function are the reconstructed/filtered
        version of the input and the coding coefficientes.
        The `states` are the coding coefficients.
        This recurrent method is the estimation process involved in
        filtering.
        """
        rho = .9
        lr = .001
        momentum = .9
        epsilon = 1e-8

        outputs = tensor.dot(states, self.W)
        rec_error = tensor.sqr(inputs - outputs).sum()
        l1_norm = tensor.sqrt(states**2 + 1e-6).sum()
        cost = rec_error + .1 * l1_norm
        grads = tensor.grad(cost,states)

        new_accum_1 = rho * accum_1 + (1 - rho) * grads**2
        new_accum_2 = momentum * accum_2 - lr * grads/ tensor.sqrt(
                new_accum_1 + epsilon)
        new_states = states + momentum * new_accum_2 - lr * (grads /
                tensor.sqrt(new_accum_1 + epsilon))
        results = [outputs, new_states, new_accum_1, new_accum_2]

        return results
