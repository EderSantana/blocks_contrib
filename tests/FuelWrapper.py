
# coding: utf-8

# In[18]:

from fuel.datasets import Dataset
from librnn.pylearn2.datasets.music import MusicSequence
from blocks.bricks import Sigmoid, Tanh, MLP, Linear, Rectifier
from blocks.bricks.recurrent import SimpleRecurrent, GatedRecurrent, LSTM
from blocks.bricks import recurrent
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Scale, Adam
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from theano import tensor
from blocks.bricks import WEIGHT, BIAS
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing
#from blocks.extensions.saveload import SerializeMainLoop
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.bricks.cost import BinaryCrossEntropy
from blocks_contrib.bricks.recurrent import DelayLine


# In[19]:

class MusicFuel(Dataset):
    def __init__(self, which_set, which_dataset):
        self.pylearn2_dataset = MusicSequence(which_set=which_set, which_dataset=which_dataset)
        self.sources = self.pylearn2_dataset.get_data_specs()[-1]

    def open(self):
        num_examples = self.pylearn2_dataset.get_num_examples()
        return self.pylearn2_dataset.iterator(1, num_examples, mode='sequential',
                                   data_specs=self.pylearn2_dataset.get_data_specs(), return_tuple=True)

    def get_data(self,state=None,request=None):
        return next(state)


# In[17]:

import theano

x, y = tensor.tensor3s('features', 'targets')

def rnn_output(x, name):
    inputtostate = Linear(name=name+'_input_to_state', input_dim=96,
                            output_dim=48)
    inputtogate  = Linear(name=name+'_input_to_gate' , input_dim=96,
                            output_dim=48)
    inputtoreset = Linear(name=name+'_input_to_reset', input_dim=96,
                            output_dim=48)
    x_s = inputtostate.apply(x)
    x_g = inputtogate.apply(x)
    x_r = inputtoreset.apply(x)

    RNN = GatedRecurrent(activation=Tanh(), dim=48,
            name=name+'_RNN', use_update_gate=True, use_reset_gate=True)
    s = RNN.apply(x_s, update_inputs=x_g, reset_inputs=x_r)

    statetooutput = Linear(name=name+'_state_to_output', input_dim=48,
                             output_dim=96)
    pre_out = statetooutput.apply(s)
    y_hat = Sigmoid(name=name+'_last_layer').apply(pre_out)
    y_hat.name = 'output_sequence'

    inputto = [inputtostate, inputtogate,
            inputtoreset, statetooutput]

    for i in inputto:
        i.weights_init = IsotropicGaussian(0.01)
        i.biases_init = Constant(0.)
    RNN.weights_init = Orthogonal()
    RNN.biases_init = Constant(0.)


    RNN.initialize()
    statetooutput.initialize()
    inputtostate.initialize()
    inputtogate.initialize()
    inputtoreset.initialize()

    return y_hat

y_hat = rnn_output(x, 'a')

predict = theano.function(inputs = [x, ], outputs = y_hat)
#cost = BinaryCrossEntropy().apply(y, y_hat)
cost = tensor.nnet.binary_crossentropy(y_hat, y).sum(axis=2).mean()
cost.name = 'BCE'

cg = ComputationGraph(cost)
params = VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)

# In[7]:

trainset = DataStream(MusicFuel(which_set='train', which_dataset='jsb'))
testset = DataStream(MusicFuel(which_set='test', which_dataset='jsb'))
validset = DataStream(MusicFuel(which_set='valid', which_dataset='jsb'))
batch_size = 1
num_epochs = 100
cost.name = "sequence_log_likelihood"
algorithm = GradientDescent(
                cost=cost, params=params,
                step_rule=Adam(0.001))
main_loop = MainLoop(
                algorithm=algorithm,
                data_stream=trainset,
                model=None,
                extensions=[FinishAfter(after_n_epochs=num_epochs),
                            TrainingDataMonitoring([cost], prefix="train",
                                                    after_every_epoch=True),
                            DataStreamMonitoring([cost], validset, prefix="valid"),
                            DataStreamMonitoring([cost], testset, prefix="test"),
                            Printing()])
main_loop.run()

# In[20]:
