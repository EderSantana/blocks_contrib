from theano import tensor
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.bricks import Initializable
from blocks.bricks.base import lazy, application
from blocks.utils import shared_floatx_nans


class BatchNorm(Initializable):
    '''Batch Normalization layer

    Although blocks already have a method for doing batch normalization,
    this one is different, it simply assumes an extra layer inside your
    MLP

    References
    ----------

    Keras neural net lib with Theano

    .. [1] https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py
    '''
    @lazy(allocation=['dim'])
    def __init__(self, dim, momentum=.9):
        self.dim = dim
        self.momentum = momentum
        self.running_mean = None
        self.running_std = None

    @property
    def W(self):
        return self.params[0]

    @property
    def b(self):
        return self.params[1]

    def _allocate(self):
        W = shared_floatx_nans((self.dim,), name='W')
        add_role(W, WEIGHT)
        self.params.append(W)
        b = shared_floatx_nans((self.dim,), name='b')
        add_role(b, BIAS)
        self.params.append(b)

    def _initialize(self):
        W, b = self.params
        self.biases_init.initialize(b, self.rng)
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, train=True):
        if train:
            m = input_.mean(axis=0)
            std = tensor.mean((input_ - m)**2 + self.epsilon, axis=0)**.5
            X = (input_ - m)/(std + self.epsilon)
            if self.running_mean is None:
                self.running_mean = m
                self.running_std = std
            else:
                self.running_mean *= self.momentum
                self.running_mean += (1-self.momentum) * m
                self.running_std *= self.momentum
                self.running_std += (1-self.momentum) * std
        else:
            X = (input_ - self.running_mean) / (self.running_std + self.epsilon)

        output = self.W * X + self.b
        return output
