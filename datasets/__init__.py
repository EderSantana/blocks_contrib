import numpy as np
import theano

from skimage.transform import rotate
floatX = theano.config.floatX


class Meanizer():
    '''Removes the mean of the datastream
    '''
    def __init__(self, datastream, indexes=[0, ]):
        self.means = []
        self.indexes = indexes
        for b in datastream.get_epoch_iterator():
            for i in indexes:
                self.means.append(b[i] * 0.)
            break

        for N, b in enumerate(datastream.get_epoch_iterator()):
            for i in indexes:
                self.means[i] += b[i]
        for i in indexes:
            self.means[i] /= (N+1)

    def meanless(self, data):
        new_data = [data[i]-self.means[i] if i in self.indexes else data[i] for i in range(len(data))]
        new_data = tuple(new_data)
        return new_data


'''Helper functions to generate rotating videos from images '''


def _allrotations(image, N):
    angles = np.linspace(0, 350, N)
    R = np.zeros((N, 784))
    for i in xrange(N):
        img = rotate(image, angles[i])
        R[i] = img.flatten()
    return R


def rotated_dataset(n_steps):
    '''Generate rotated images.
    This functions is supposed to be used with `fuel.transformer.Mapping`
    '''
    def func(data):
        newfirst = data[0]
        Rval = np.zeros((n_steps, newfirst.shape[0], newfirst.shape[1]))
        for i, sample in enumerate(newfirst):
            Rval[:, i, :] = _allrotations(sample.reshape((28, 28)), n_steps)
        Rval = Rval.astype(floatX)
        return (Rval, data[1])
    return func
