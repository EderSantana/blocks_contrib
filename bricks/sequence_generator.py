from theano import tensor
from blocks.base import lazy, application
from blocks.bricks import Initializable
from blocks.bricks.sequence_generators import TrivialEmitter, TrivialFeedback


class MLPEmitter(TrivialEmitter, Initializable):
    """A generic MLP emitter with binary crosentropy cost

    Parameters
    ----------
    initial_output : int or a scalar :class:`~theano.Variable`
        The initial output.
    mlp : Brick :class:`bricks.MLP`

    """
    @lazy
    def __init__(self, mlp=None, **kwargs):
        self.mlp = mlp
        super(MLPEmitter, self).__init__(**kwargs)
        self.children = [mlp,]

    @application
    def emit(self, readouts):
        return self.mlp.apply(readouts)
        #return readouts

    @application
    def cost(self, readouts, outputs):
        # the next two clips are for sanity reasons only
        outputs  = tensor.clip(outputs, 0, 1)
        readouts = tensor.clip(readouts,0, 1)
        return tensor.nnet.binary_crossentropy(readouts,
                  outputs).sum(axis=readouts.ndim-1)

    #@application
    #def initial_outputs(self, batch_size, *args, **kwargs):
    #    return tensor.zeros((batch_size, self.mlp.output_dim))

    def get_dim(self, name):
        if name == 'outputs':
            return self.mlp.output_dim
        return super(MLPEmitter, self).get_dim(name)


class MLPFeedback(TrivialFeedback, Initializable):
    """A generica MLP feedback

    Parameters
    ----------

    mlp : Brick :class:`bricks.MLP`
        defines the transformation from output back to hidden state
    """
    @lazy
    def __init__(self, mlp=None, **kwargs):
        self.mlp = mlp
        kwargs['ouput_dim'] = mlp.output_dim
        super(MLPFeedback, self).__init__(**kwargs)
        self.children = [mlp]

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return self.mlp.apply(outputs)