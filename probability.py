'''Helper functions for calculating probability matrices
'''
import numpy as np
import h5py
import theano

from scipy.spatial.distance import squareform
from sklearn.manifold.t_sne import _joint_probabilities
from sklearn.metrics.pairwise import pairwise_distances
floatX = theano.config.floatX


def get_probability_matrices(datastream, num_batches, h5path, perplexity=30):
    '''Get joint probability matrices

    Uses sklearn _joint_probabilities function to all calculate the probability
    matrices of all the batches of a `fuel.datastream`. This function assumes that
    the data of interest is the first elemeent of each batch. Also it assumes
    that the data was already transposed to the form time x batch x dimension.
    '''
    h5obj = h5py.File(h5path)
    dt = h5py.special_dtype(vlen=np.dtype(floatX))
    d = h5obj.create_dataset('probability_matrices', (num_batches, ), dtype=dt)
    for i, b in enumerate(datastream.get_epoch_iterator()):
        D = pairwise_distances(b[0], metric='euclidean', squared=True)
        P = _joint_probabilities(D.astype('float64'), perplexity, verbose=False)
        d[i] = P
    h5obj.close()


def probability_matrices_generator(h5path, num_batches):
    i = -1
    h5obj = h5py.File(h5path, 'r')
    P = h5obj['probability_matrices']
    while True:
        i += 1
        idx = i % num_batches
        yield squareform(P[idx])
    h5obj.close()
