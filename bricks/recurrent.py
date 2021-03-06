from theano import tensor
from theano.scan_module import until

from blocks.bricks import Initializable
from blocks.bricks.base import lazy, application
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans
from blocks.bricks.recurrent import BaseRecurrent, recurrent


class ConditionedRecurrent(BaseRecurrent):
    """ConditionedRecurrent network

    A recurrent network that unfolds an input vector to a sequence.

    Parameters
    ----------
    wrapped : instance of :class:`BaseRecurrent`
        A brick that will get the input vector.

    Notes
    -----
    See :class:`.BaseRecurrent` for initialization parameters.

    """
    def __init__(self, wrapped, **kwargs):
        super(ConditionedRecurrent, self).__init__(**kwargs)
        self.wrapped = wrapped
        self.children = [wrapped, ]

    def get_dim(self, name):
        if name == 'attended':
            return self.wrapped.get_dim('inputs')
        if name == 'attended_mask':
            return 0
        return self.wrapped.get_dim(name)

    @application(contexts=['attended', 'attended_mask'])
    def apply(self, **kwargs):
        context = kwargs['attended']
        try:
            kwargs['inputs'] += context.dimshuffle('x', 0, 1)
        except:
            kwargs['inputs'] = context.dimshuffle('x', 0, 1)
        for context in ConditionedRecurrent.apply.contexts:
            kwargs.pop(context)
        return self.wrapped.apply(**kwargs)

    @apply.delegate
    def apply_delegate(self):
        return self.wrapped.apply


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

    @recurrent(sequences=[], states=['states'],
               outputs=['states', 'flags'],
               contexts=['inputs'])
    def apply(self, inputs=None, states=None, **kwargs):
        outputs = self.children[0].apply(inputs=inputs,
                                         states=states,
                                         iterate=False,
                                         **kwargs)
        flags = self.children[1].apply(outputs).sum()
        stop_condition = flags >= .5
        outputs = [outputs, flags]
        return outputs, until(stop_condition)
    #TODO define outputs_info, define RecurrentFlag class

class UnfolderLSTM(Initializable, BaseRecurrent):
    """UnfolderLSTM network

    A recurrent network that unfolds an input vector to a sequence.

    Parameters
    ----------
    prototype : instance of :class:`LSTM`
        A brick that will get the input vector.

    flagger : instance of :class:`Brick``
        A brick that will flag when to stop the loop

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy
    def __init__(self, prototype, flagger, **kwargs):
        super(UnfolderLSTM, self).__init__(**kwargs)
        self.children = [prototype, flagger]

    def get_dim(self, name):
        if name in ('inputs', 'states', 'cells'):
            return self.children[0].get_dim(name)
        else:
            return super(UnfolderLSTM, self).get_dim(name)

    #def initial_state(self, state_name, batch_size, *args, **kwargs):

    @recurrent(sequences=[], states=['states', 'cells'],
               outputs=['states','cells', 'flags'], contexts=['inputs'])
    def apply(self, inputs=None, states=None, cells=None, **kwargs):
        outputs = self.children[0].apply(inputs=inputs,
                                         cells=cells,
                                         states=states,
                                         iterate=False,
                                         **kwargs)
        flags = self.children[1].apply(outputs[0]).sum()
        stop_condition = flags >= .5
        outputs.append(flags)
        return outputs, until(stop_condition)

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
        '''TODO
        Delay line should support mask to handle differently
        sized sequences
        '''
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
