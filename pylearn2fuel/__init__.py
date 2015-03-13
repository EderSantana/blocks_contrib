from fuel.datasets import Dataset
import numpy as np

class Pylearn2Dataset(Dataset):
    '''Pylearn2Dataset wraps a `pylearn2.dataset` object and adds only the
    minimal `fuel` interface. An object of this class can be used as input to
    `fuel.streams.DataStream`.
    
    Parameters
    ----------
    dataset: `pylearn2.dataset` object 
        Note that this is expecting the actual the object will be initialized inside
    batch_size: int
        Batch size to be used by the `pylearn2.dataset` iterator.
    '''
    def __init__(self, dataset, batch_size, **kwargs):
        self.pylearn2_dataset = dataset
        self.sources = self.pylearn2_dataset.get_data_specs()[1]
        self.batch_size = batch_size 

    def open(self):
        num_examples = self.pylearn2_dataset.get_num_examples()
        iterator = self.pylearn2_dataset.iterator(
                   self.batch_size,
                   num_examples/self.batch_size,
                   mode='sequential',
                   data_specs=self.pylearn2_dataset.get_data_specs(),
                   return_tuple=True)
        return iterator    
    def get_data(self,state=None,request=None):
        return next(state)

class Pylearn2DatasetNoise(Dataset):
    '''Pylearn2DatasetNoise is the same as `Pylearn2Dataset` with some an 
    extra batch of random nubmer.
    
    Parameters
    ----------
    dataset: `pylearn2.dataset` object 
        Note that this is expecting the actual the object will be 
        initialized inside
    batch_size: int
        Batch size to be used by the `pylearn2.dataset` iterator.
    noise_dim: int
        Dimmension of the noise batch
    '''
    def __init__(self, dataset, batch_size, noise_dim, which_sources=[0,1], 
                **kwargs):
        self.pylearn2_dataset = dataset
        self.sources = self.pylearn2_dataset.get_data_specs()[1]
        self.sources = tuple([self.sources[i] for i in which_sources])
        self.sources = self.sources + tuple('eps')
        self.sources = ['features', 'eps']
        self.batch_size = batch_size 
        self.noise_dim = noise_dim
        self.which_sources = which_sources
    def open(self):
        num_examples = self.pylearn2_dataset.get_num_examples()
        iterator = self.pylearn2_dataset.iterator(
                   self.batch_size,
                   num_examples/self.batch_size,
                   mode='sequential',
                   data_specs=self.pylearn2_dataset.get_data_specs(),
                   return_tuple=True)
        return iterator    
    def get_data(self,state=None,request=None):
        batch = next(state)
        batch = [batch[i] for i in self.which_sources]
        eps = np.random.normal(0,1,size=(batch[0].shape[0], 
                               batch[0].shape[1], self.noise_dim))
        batch = tuple(batch + list(eps))
        return batch
