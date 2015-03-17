from theano import tensor
from theano.scan_module import until

from blocks.bricks import Initializable
from blocks.bricks.base import lazy, application
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans
from blocks.bricks.recurrent import BaseRecurrent, recurrent

#class GammaRecurrent(SimpleRecurrent):

class Unfolder(Initializable, BaseRecurrent):
    """Unfolder network

    A recurrent network that unfolds an input vector to a sequence.

    Parameters
    ----------
    prototype : instance of :class:`BaseRecurrent`
        A brick that will get the input vector.

    flagger : instance of :class:`Brick``
        A brick that will flag when to stop the loop

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy
    def __init__(self, prototype, flagger, **kwargs):
        super(Unfolder, self).__init__(**kwargs)
        self.children = [prototype, flagger]

    def get_dim(self, name):
        if name in ('inputs', 'states'):
            return self.children[0].dim
        else:
            return super(Unfolder, self).get_dim(name)

    #def initial_state(self, state_name, batch_size, *args, **kwargs):

    @recurrent(sequences=[], states=['states'], outputs=['states'],
               contexts=['inputs'])
    def apply(self, inputs=None, states=None):
        outputs = self.children[0].apply(inputs=inputs, states=states,
                                         iterate=False)

        flags = self.children[1].apply(outputs).sum()
        stop_condition = flags >= .5
        return outputs, until( stop_condition )
    #TODO define outputs_info, define RecurrentFlag class

class DelayLine(BaseRecurrent, Initializable):
    """Store and (optionally transform) previous inputs in a delay line
    """
    @lazy
    def __init__(self, input_dim, memory_size, fixed=False,
                 **kwargs):
        super(DelayLine, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.fixed = fixed
        self.output_dim = input_dim * memory_size

    @property
    def W(self):
        return self.mu

    def _allocate(self):
        self.mu = shared_floatx_nans((self.output_dim-self.input_dim),
                name='mu')
        add_role(self.mu, WEIGHT)
        if not self.fixed:
            self.params.append(self.mu)
        '''
        self.delay_line = shared_floatx_nans((self.batch_size,
                                              self.input_dim*self.memory_size),
                                              'delay_line')
        '''

    def _initialize(self):
        mu, = self.params
        self.weights_init.initialize(mu, self.rng)
        #self.biases_init.initialize(self.delay_line, self.rng)

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        return tensor.zeros((batch_size, self.output_dim))

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs=None, states=None, mask=None):
        mu, = self.params
        mu = tensor.clip(mu, -1., 1.)
        m_new = states[:,:-self.input_dim]
        m_prev  = states[:,self.input_dim:]
        m_act = (1.-mu)*m_prev + mu*m_new
        next_states = tensor.concatenate((inputs, m_act), axis=-1)
        return next_states

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return self.output_dim
        if name in (DelayLine.apply.sequences +
                    DelayLine.apply.states):
            return self.output_dim
        super(DelayLine, self).get_dim(name)
