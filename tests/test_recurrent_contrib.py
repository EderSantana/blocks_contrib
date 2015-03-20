import numpy as np
import theano
import logging

from numpy.testing import assert_allclose
from theano import tensor, function

from blocks import initialization
from blocks.bricks import Identity, Linear
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant

from blocks_contrib.bricks.recurrent import DelayLine
from blocks_contrib.bricks.recurrent import Unfolder, UnfolderLSTM

logger = logging.getLogger(__name__)

def test_constant_input_lstm():
    x = tensor.matrix('x')
    proto = LSTM(activation=Identity(), dim=1,
                weights_init=Constant(1/4.),
                biases_init=Constant(0.))
    proto.initialize()

    flagger = Linear(input_dim=1, output_dim=1,
                     weights_init=Constant(1./2.),
                     biases_init=Constant(0.))
    flagger.initialize()

    inp2hid = Linear(input_dim=1, output_dim=4,
                     weights_init=Constant(1/4.),
                     biases_init=Constant(0))
    inp2hid.initialize()

    rnn = UnfolderLSTM(proto, flagger)
    rnn.initialize()

    h = inp2hid.apply(x)
    y = rnn.apply(inputs=h, n_steps=10, batch_size=5)

    F = function([x],y)
    X = np.ones((5,1)).astype(theano.config.floatX)

    H = function([x], flagger.apply(x))
    T = H(X)
    #print T

    Y = F(X)
    print Y
    print Y[0].shape

    assert Y[0].shape == (4,5,1)
    #target = np.cumsum(np.ones((6,1,1)),axis=0)
    #assert_allclose(Y, target)

def test_constant_input():
    x = tensor.matrix('x')
    proto = SimpleRecurrent(activation=Identity(), dim=1,
            weights_init=initialization.Identity(1.))
    proto.initialize()

    flagger = Linear(input_dim=1, output_dim=1,
                     weights_init=Constant(1/10.),
                     biases_init=Constant(0.))
    flagger.initialize()

    rnn = Unfolder(proto, flagger)
    rnn.initialize()

    y = rnn.apply(inputs=x, n_steps=10, batch_size=1)

    F = function([x],y)
    X = np.ones((1,1)).astype(theano.config.floatX)

    H = function([x], flagger.apply(x))
    T = H(X)
    print T

    Y = F(X)
    print Y

    target = np.cumsum(np.ones((5,1,1)),axis=0)
    assert_allclose(Y[0], target)

def test_delay_line():
    x = tensor.tensor3('x')
    input_dim = 1
    batch_size = 1
    memory_size = 3
    time_len = 4
    delay_line = DelayLine(input_dim, memory_size,
                           weights_init=Constant(1.))
    delay_line.initialize()
    y = delay_line.apply(x, iterate=True)
    func = function([x], y)

    x_val = np.zeros((time_len, batch_size, input_dim))
    val = np.arange(4)
    #val = np.tile(x_val[np.newaxis], batch_size).T
    x_val[:,0,0] = val.astype(theano.config.floatX)
    x_val = x_val.astype(theano.config.floatX)
    y_val = func(x_val).astype(theano.config.floatX)
    print y_val
    assert y_val.shape == (4,1,3)

if __name__ == '__main__':
    #test_delay_line()
    test_constant_input()
    test_constant_input_lstm()
