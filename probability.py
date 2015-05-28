'''Helper functions for calculating probability matrices
'''
from __future__ import print_function
import numpy as np
import h5py
import theano

from numpy.fft import fft, ifft
from scipy.spatial.distance import squareform
from sklearn.manifold.t_sne import _joint_probabilities
from sklearn.metrics.pairwise import pairwise_distances
try:
    from numba import autojit
except:
    print("`numba` is not installed. Calculatating `_shift_inv_dist` will be slow!!!")

    def autojit(func):
        return func

floatX = theano.config.floatX


def fft_shit_inv_pairwise_distance(X):
    '''Calculates a time (axis 1) shift invariant distance matrix.

    For each pair of samples in the rows of X, this functions calculates
    a delayed displacement between the two rows that provides the smaller
    distance for all delays.
    This functions is memory hungry. Prefer `shift_inv_pairwise_distance`
    below that can be as fast, but requires Numba.

    Parameters
    ----------
    X: `numpy.array`
        tensor with 3 dimensions samples x time x dim

    Returns
    -------
    D: `numpy.array`
        sampless x samples distance matrix
    '''
    S = (X**2).sum(axis=(1, 2))
    Fx = fft(X, axis=1)
    XY = ((Fx[:, None, :, :]) * np.conj(Fx))
    xy = abs(ifft(XY, axis=(2))).max(axis=2)
    return (S[:, None] + S[None, :] - 2*xy.sum(axis=-1))


@autojit
def _time_shift_invariant_pairwise_product(X):
    b, t, d = X.shape
    D1 = np.zeros((b, b))
    larger1 = 0.
    T1 = 0.
    for h in range(d):
        for i in range(b):
            for j in range(b):
                larger1 = 0.
                for k in range(t):
                    T1 = 0.
                    for m in range(t-k):
                        T1 += X[i, m, h] * X[j, m+k, h]
                    if T1 > larger1:
                        larger1 = T1
                D1[i, j] += larger1
    return D1


def shift_inv_pairwise_distance(X):
    '''Calculates a time (axis 1) shift invariant distance matrix.

    For each pair of samples in the rows of X, this functions calculates
    a delayed displacement between the two rows that provides the smaller
    distance for all delays.
    It better to have Numba installed for speed up

    Parameters
    ----------
    X: `numpy.array`
        tensor with 3 dimensions samples x time x dim

    Returns
    -------
    D: `numpy.array`
        sampless x samples distance matrix
    '''
    S = (X**2).sum(axis=(1, 2))
    xy = _time_shift_invariant_pairwise_product(X)
    return S[:, None] + S[None, :] - xy - xy.T


def get_shift_inv_probability_matrices(datastream, num_batches, h5path,
                                       shift_inv_pw=shift_inv_pairwise_distance,
                                       perplexity=30):
    '''Get joint probability matrices using a shift invariant metric

    Uses sklearn _joint_probabilities function to all calculate the probability
    matrices of all the batches of a `fuel.datastream`. This function assumes that
    the data of interest is the first elemeent of each batch. Also it assumes
    that the data was already transposed to the form time x batch x dimension.

    Parameters
    ----------
    datastream: `fuel.datastream`
    num_batches: int
        total number of batches
    h5path: str
        path to save the hdf5 file at
    shift_inv_pw: function (default: shift_inv_pairwise_distance)
        a function to calculate pairwise distance matrices
    perplexity: int > 0
        perplexity of the probability distributions, which is approximately
        the number of neighbors each sample has
    '''
    h5obj = h5py.File(h5path)
    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    d = h5obj.create_dataset('probability_matrices', (num_batches, ), dtype=dt)
    for i, b in enumerate(datastream.get_epoch_iterator()):
        print('Processing batch number {} / {}'.format(i+1, num_batches))
        D = shift_inv_pw(b[0].transpose(1, 0, 2))
        P = _joint_probabilities(D.astype('float64'), perplexity, verbose=False)
        d[i] = P.astype('float32')
    h5obj.close()


def get_probability_matrices(datastream, num_batches, h5path, perplexity=30):
    '''Get joint probability matrices

    Uses sklearn _joint_probabilities function to all calculate the probability
    matrices of all the batches of a `fuel.datastream`. This function assumes that
    the data of interest is the first elemeent of each batch. Also it assumes
    that the data was already transposed to the form time x batch x dimension.

    Parameters
    ----------
    datastream: `fuel.datastream`
    num_batches: int
        total number of batches
    h5path: str
        path to save the hdf5 file at
    perplexity: int > 0
        perplexity of the probability distributions, which is approximately
        the number of neighbors each sample has
    '''
    h5obj = h5py.File(h5path)
    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    d = h5obj.create_dataset('probability_matrices', (num_batches, ), dtype=dt)
    for i, b in enumerate(datastream.get_epoch_iterator()):
        D = pairwise_distances(b[0], metric='euclidean', squared=True)
        P = _joint_probabilities(D.astype('float64'), perplexity, verbose=False)
        d[i] = P.astype('float32')
    h5obj.close()


class Pserver():
    '''Probability matrices server
    Loads an hdf5 and server a row a time. The fuel_next_P method of this class
    is supposed to be used with a 'fuel.transformer.Mapping' functions, using
    a add_source parameter

    '''
    def __init__(self, h5path, num_batches, start=0):
        self.num_batches = num_batches
        self.start = start
        self.h5obj = h5py.File(h5path, 'r')
        self.counter = -1

    def get_next_P(self, data):
        self.counter += 1
        idx = self.counter % self.num_batches
        idx += self.start
        P = self.h5obj['probability_matrices'][idx]
        P = squareform(P).astype(floatX)
        P = np.maximum(P, 1e-12)
        return (P.astype('float32'), )

    def close(self):
        self.h5obj.close()


def probability_matrices_generator(h5path, num_batches):
    '''Similar to `class:Pserver`, but simpler and less usefull.
    '''
    i = -1
    h5obj = h5py.File(h5path, 'r')
    P = h5obj['probability_matrices']
    while True:
        i += 1
        idx = i % num_batches
        yield squareform(P[idx])
    h5obj.close()
