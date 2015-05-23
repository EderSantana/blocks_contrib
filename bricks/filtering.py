import numpy
import theano

from theano import tensor
from blocks.bricks import Initializable
from blocks.bricks.recurrent import recurrent, BaseRecurrent
from blocks.bricks.base import lazy, application
from blocks.graph import ComputationGraph

from blocks_contrib.utils import diff_abs, l2_norm_cost
floatX = theano.config.floatX


def RMSPropStep(cost, states, accum_1, accum_2):
    rho = .9
    lr = .001
    momentum = .9
    epsilon = 1e-8

    grads = tensor.grad(cost, states)

    new_accum_1 = rho * accum_1 + (1 - rho) * grads**2
    new_accum_2 = momentum * accum_2 - lr * grads / tensor.sqrt(new_accum_1 + epsilon)
    new_states = states + momentum * new_accum_2 - lr * (grads /
                                                         tensor.sqrt(new_accum_1 + epsilon))
    return new_states, new_accum_1, new_accum_2


class SparseFilter(BaseRecurrent, Initializable):
    def __init__(self, mlp, *args, **kwargs):
        super(SparseFilter, self).__init__(*args, **kwargs)
        self.mlp = mlp
        self.children = [mlp, ]

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        if state_name in self.apply.states:
            dim = self.get_dim(state_name)
            return tensor.zeros((batch_size, dim))
        if state_name == 'gamma':
            dim = self.get_dim('gamma')
            return .1*tensor.ones((batch_size, dim))
        return super(SparseFilter, self).initial_state(state_name,
                                                       batch_size, *args, **kwargs)

    def get_dim(self, name):
        if name in (self.apply.states +
                    self.apply.outputs[1:]) + ['prior', 'gamma', 'outputs']:
            return self.mlp.input_dim
        elif name == 'inputs':
            return self.mlp.output_dim
        return super(SparseFilter, self).get_dim(name)

    @recurrent(sequences=[], states=['states', 'accum_1', 'accum_2'],
               outputs=['outputs', 'states',
                        'accum_1', 'accum_2'],
               contexts=['inputs', 'prior', 'gamma'])
    def apply(self, inputs=None,
              states=None, accum_1=None,
              accum_2=None, gamma=.1, prior=None):
        """ The outputs of this function are the reconstructed/filtered
        version of the input and the coding coefficientes.
        The `states` are the coding coefficients.
        This recurrent method is the estimation process involved in
        filtering.

        """
        if prior is not None:
            cost = .01 * diff_abs(states - prior).sum()
        else:
            cost = 0
        outputs = self.mlp.apply(states)
        # TODO accept `blocks.bricks.cost` as input to be used as reconstruction cost
        rec_error = tensor.sqr(inputs - outputs).sum()
        l1_norm = (gamma * diff_abs(states)).sum()
        cost += rec_error + l1_norm
        new_states, new_accum_1, new_accum_2 = RMSPropStep(cost, states,
                                                           accum_1, accum_2)
        results = [outputs, new_states, new_accum_1, new_accum_2]
        return results

    @application
    def cost(self, inputs, n_steps, batch_size, gamma=.1, prior=None):
        z = self.apply(inputs=inputs, gamma=gamma, prior=prior, n_steps=n_steps,
                       batch_size=batch_size)[1][-1]
        z = theano.gradient.disconnected_grad(z)
        # x_hat = tensor.dot(z, self.W)
        x_hat = self.mlp.apply(z)
        cost = tensor.sqr(inputs - x_hat).sum()
        weights_normalization = l2_norm_cost(self.mlp, ComputationGraph([cost]), .01)
        cost += weights_normalization
        return cost, z, x_hat


class VarianceComponent(SparseFilter):
    def __init__(self, mlp, *args, **kwargs):
        super(VarianceComponent, self).__init__(mlp, *args, **kwargs)

    def get_dim(self, name):
        if name == 'prev_code':
            return self.mlp.input_dim
        if name == 'prior':
            return self.mlp.outputdim
        return super(VarianceComponent, self).get_dim(name)

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        if state_name in self.apply.states + ['prior', 'prev_code']:
            dim = self.get_dim('states')
            return tensor.zeros((batch_size, dim))
        if state_name == 'prev_rec':
            dim = self.get_dim('input')
            return tensor.zeros((batch_size, dim))
        return super(VarianceComponent, self).initial_state(state_name,
                                                            batch_size, *args, **kwargs)

    @recurrent(sequences=[], states=['states', 'accum_1', 'accum_2'],
               outputs=['outputs', 'states',
                        'accum_1', 'accum_2'],
               contexts=['prior', 'prev_rec', 'prev_code'])
    def apply(self, states=None, accum_1=None,
              accum_2=None, batch_size=None, prior=None,
              prev_rec=None, prev_code=None):
        """ The outputs of this function are the higher order
        variance components.

        The `states` are the coding coefficients.
        This recurrent method is the estimation process involved in
        filtering.

        """
        if prior is not None:
            cost = .01 * diff_abs(states - prior).sum()
        else:
            cost = 0
        # uW = self.mlp.apply(states)
        # outputs = .05 * (1 + tensor.exp(-uW))
        # outputs = .1 * tensor.nnet.sigmoid(uW)
        outputs = self.get_sparseness(states)
        prev_l1 = (outputs * diff_abs(prev_code)).sum()
        # rec_error = tensor.sqr(inputs - prev_rec).sum()
        l1_norm = diff_abs(states).sum()
        cost += prev_l1 + .1 * l1_norm
        new_states, new_accum_1, new_accum_2 = RMSPropStep(cost, states,
                                                           accum_1, accum_2)
        results = [outputs, new_states, new_accum_1, new_accum_2]
        return results

    @application
    def get_sparseness(self, u):
        uW = self.mlp.apply(u)
        return .05 * (1 + tensor.exp(-uW))

    @application
    def cost(self, prev_code, prior=None):
        u = self.apply(batch_size=self.batch_size, prev_code=prev_code,
                       n_steps=self.n_steps, prior=prior)[1][-1]
        u = theano.gradient.disconnected_grad(u)
        # uW = self.mlp.apply(u)
        # outputs = .05 * (1 + tensor.exp(-uW))
        outputs = self.get_sparseness(u)
        # outputs = .1 * tensor.nnet.sigmoid(uW)
        final_cost = (outputs*prev_code).sum() + .01*tensor.sqr(self.W).sum()
        return final_cost, u, outputs


class TemporalSparseFilter(BaseRecurrent, Initializable):
    def __init__(self, proto, transition, n_steps, batch_size, *args, **kwargs):
        super(TemporalSparseFilter, self).__init__(*args, **kwargs)
        self.proto = proto
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.transition = transition
        self.children = [proto, proto.mlp, transition]

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        return self.proto.initial_state(state_name,
                                        batch_size, *args, **kwargs)

    def get_dim(self, name):
        return self.proto.get_dim(name)

    @recurrent(sequences=['inputs'], states=['states'],
               outputs=['outputs', 'states'],
               contexts=[])
    def apply(self, inputs=None, states=None, **kwargs):
        """ The outputs of this function are the reconstructed/filtered
        version of the input and the coding coefficientes.
        The `states` are the coding coefficients.
        This recurrent method is the estimation process involved in
        filtering.

        """
        prior = theano.gradient.disconnected_grad(states)
        prior = self.transition.apply(states)
        results = self.proto.apply(inputs=inputs, prior=prior,
                                   n_steps=self.n_steps, batch_size=self.batch_size, **kwargs)
        return results[0][-1], results[1][-1]

    @application
    def cost(self, inputs, **kwargs):
        x_hat, z = self.apply(inputs=inputs, **kwargs)
        z = theano.gradient.disconnected_grad(z)
        prev = self.transition.apply(z)
        innovation_error = .01 * diff_abs(z[1:] - prev[:-1]).sum()
        x_hat = self.proto.mlp.apply(z)
        main_cost = tensor.sqr(inputs - x_hat).sum() + innovation_error
        cg = ComputationGraph([main_cost])
        weights_normalization = l2_norm_cost(self.proto.mlp, cg, .01)
        weights_normalization += l2_norm_cost(self.transition, cg, .01)
        costs = main_cost + weights_normalization
        return costs, z, x_hat


class TemporalVarComp(BaseRecurrent, Initializable):
    def __init__(self, slayer, stransition, clayer, n_steps, batch_size, *args, **kwargs):
        '''
        Paramters
        ---------
        slayer: `SparseFilter`
            states layer, does sparse coding
        stransition: `bricks.MLP`
            transition function of the sparse coding
        stransition: `VarianceComponent`
            causes layers, does variance component learning

        '''
        super(TemporalVarComp, self).__init__(*args, **kwargs)
        self.slayer = slayer
        self.clayer = clayer

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.stransition = stransition
        self.children = [slayer, slayer.mlp, clayer, clayer.mlp, stransition]

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        if state_name == 'sstates':
            return self.slayer.initial_state('states',
                                             batch_size, *args, **kwargs)
        if state_name == 'cstates':
            return self.clayer.initial_state('states',
                                             batch_size, *args, **kwargs)

    def get_dim(self, name):
        if name == 'sstates':
            return self.slayer.get_dim('states')
        elif name == 'cstates':
            return self.clayer.get_dim('states')

    @recurrent(sequences=['inputs'], states=['sstates', 'cstates'],
               outputs=['soutputs', 'sstates', 'coutputs', 'cstates'],
               contexts=[])
    def apply(self, inputs=None, sstates=None, cstates=None, **kwargs):
        """ The outputs of this function are the reconstructed/filtered
        version of the input and the coding coefficientes.
        The `states` are the coding coefficients.
        This recurrent method is the estimation process involved in
        filtering.

        """
        sprior = theano.gradient.disconnected_grad(sstates)
        sprior = self.stransition.apply(sstates)
        cprior = theano.gradient.disconnected_grad(cstates)
        gamma = self.clayer.get_sparseness(cprior)
        sparse_code = self.slayer.apply(inputs=inputs, prior=sprior, gamma=gamma,
                                        n_steps=self.n_steps, batch_size=self.batch_size)
        variance_code = self.clayer.apply(prior=cprior, prev_code=sparse_code[1][-1],
                                          n_steps=self.n_steps, batch_size=self.batch_size, **kwargs)
        return sparse_code[0][-1], sparse_code[1][-1], variance_code[0][-1], variance_code[1][-1]

    @application
    def cost(self, inputs, **kwargs):
        x_hat, z, gammas, u = self.apply(inputs=inputs, **kwargs)
        z = theano.gradient.disconnected_grad(z)
        u = theano.gradient.disconnected_grad(z)
        prev = self.stransition.apply(z)
        innovation_error = .01 * diff_abs(z[1:] - prev[:-1]).sum()
        x_hat = self.slayer.mlp.apply(z)
        sparseness = (self.clayer.get_sparseness(u) * diff_abs(z)).sum()
        main_cost = tensor.sqr(inputs - x_hat).sum() + innovation_error + sparseness
        cg = ComputationGraph([main_cost])
        weights_normalization = l2_norm_cost(self.slayer.mlp, cg, .01)
        weights_normalization += l2_norm_cost(self.stransition, cg, .01)
        weights_normalization += l2_norm_cost(self.clayer.mlp, cg, .01)
        costs = main_cost + weights_normalization
        return costs, z, x_hat, u


class VariationalSparseFilter(SparseFilter):
    @lazy(allocation=['dim', 'input_dim', 'batch_size', 'n_steps'])
    def __init__(self, mlp, *args, **kwargs):
        super(VariationalSparseFilter, self).__init__(*args, **kwargs)
        self.mlp = mlp
        self.children = [self.mlp, ]

    @recurrent(sequences=['noise'], states=['states_mean', 'accum_1_m', 'accum_2_m',
                                            'states_log_sigma', 'accum_1_ls', 'accum_2_ls'],
               outputs=['outputs', 'codes', 'states_mean',
                        'accum_1_m', 'accum_2_m', 'states_log_sigma', 'accum_1_ls', 'accum_2_ls'],
               contexts=['inputs', 'prior', 'gamma'])
    def apply(self, noise=None, inputs=None,
              states_mean=None, states_log_sigma=None, accum_1_m=None,
              accum_2_m=None, accum_1_ls=None, accum_2_ls=None, gamma=.1, prior=None):
        """ The outputs of this function are the reconstructed/filtered
        version of the input and the coding coefficientes.
        The `states` are the coding coefficients.
        This recurrent method is the estimation process involved in
        filtering.

        """
        sigma = tensor.exp(states_log_sigma)
        z = states_mean + noise * sigma
        if prior is not None:
            tstates = tensor.dot(z, tensor.eye(self.dim))
            cost = .01 * diff_abs(tstates - prior).sum()
        else:
            cost = 0
        tinputs = tensor.dot(inputs, tensor.eye(self.input_dim))
        outputs = self.mlp.apply(z)  # tensor.dot(z, self.W)
        rec_error = tensor.sqr(tinputs - outputs).sum()
        l1_mean = diff_abs(states_mean)
        l1_sigma = diff_abs(sigma - 1)
        l1_norm = (gamma * (l1_mean + l1_sigma)).sum()
        cost += rec_error + l1_norm
        new_means_stuff = RMSPropStep(cost, states_mean, accum_1_m, accum_2_m)
        new_log_sigma_stuff = RMSPropStep(cost, states_log_sigma, accum_1_ls, accum_2_ls)
        results = (outputs, z) + new_means_stuff + new_log_sigma_stuff
        return results

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        if state_name in self.apply.states:
            dim = self.dim
            # zeros = numpy.zeros((self.batch_size, dim))
            # return theano.shared(zeros.astype(floatX))
            return tensor.zeros((batch_size, dim))
        if state_name == 'gamma':
            dim = self.get_dim('gamma')
            gammas = .1*numpy.ones((self.batch_size, dim))
            return theano.shared(gammas.astype(floatX))
        return super(VariationalSparseFilter, self).initial_state(state_name,
                                                                  batch_size, *args, **kwargs)

    @application
    def cost(self, inputs, noise, gamma=.1, prior=None):
        z = self.apply(noise=noise, inputs=inputs, gamma=gamma, prior=prior)[1][-1]
        z = theano.gradient.disconnected_grad(z)
        x_hat = self.mlp.apply(z)  # tensor.dot(z, self.W)
        cost = tensor.sqr(inputs - x_hat).sum()  # + .01*tensor.sqr(self.W).sum()
        cost += l2_norm_cost(self.mlp, ComputationGraph([cost]), .01)
        return cost, z, x_hat
