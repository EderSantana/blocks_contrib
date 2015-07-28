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


def _allrotations(image, N, img_shape=(28, 28), final_angle=350):
    angles = np.linspace(0, final_angle, N)
    R = np.zeros((N, np.prod(img_shape)))
    for i in xrange(N):
        img = rotate(image, angles[i])
        if len(img_shape) == 3:
            img = img.transpose(2, 0, 1)
        R[i] = img.flatten()
    return R


def diff_length_rotated(max_steps, min_steps, img_shape=(28, 28), final_angle=350):
    def func(data):
        newfirst = data[0]
        Rval = [list() for i in range(newfirst.shape[0])]
        for i, sample in enumerate(newfirst):
            if len(img_shape) == 3:
                I = sample.reshape(img_shape).transpose(1, 2, 0)
            else:
                I = sample.reshape(img_shape)
            n_steps = np.random.randint(min_steps, max_steps)
            Rval[i] = _allrotations(I, n_steps, img_shape, final_angle)
            ret = (Rval, ) + data[1:]
            return ret


def rotated_dataset(n_steps, img_shape=(28, 28), final_angle=350):
    '''Generate rotated images.
    This functions is supposed to be used with `fuel.transformer.Mapping`
    '''
    def func(data):
        newfirst = data[0]
        Rval = np.zeros((n_steps, newfirst.shape[0], newfirst.shape[1]))
        for i, sample in enumerate(newfirst):
            if len(img_shape) == 3:
                I = sample.reshape(img_shape).transpose(1, 2, 0)
            else:
                I = sample.reshape(img_shape)
            Rval[:, i, :] = _allrotations(I, n_steps, img_shape, final_angle)
        Rval = Rval.astype(floatX)
        ret = (Rval, ) + data[1:]
        return ret
    return func
