import numpy as np
import theano

from theano import tensor

from blocks.bricks.base import application
from blocks.bricks.cost import Cost

from blocks_contrib.utils import distance_matrix, zero_diagonal, l2
floatX = theano.config.floatX


class GaussianPrior(Cost):
    @application(outputs=['cost'])
    def apply(self, mean, log_sigma, prior_mean=0, prior_log_sigma=0):
        kl = (prior_log_sigma - log_sigma + 0.5 * (tensor.exp(2 * log_sigma) +
                                                   (mean - prior_mean) ** 2
                                                   ) / tensor.exp(2 * prior_log_sigma) - 0.5
              ).sum(axis=-1)
        return kl


class GaussianMSE(Cost):
    @application(output=['cost'])
    def apply(self, y, mean=0, log_sigma=1):
        logpxz = ((0.5 * np.log(2 * np.pi) + log_sigma) +
                  0.5 * ((y - mean) / tensor.exp(log_sigma))**2).sum(axis=-1)
        return logpxz


class tSNE(Cost):
    @application(output=['cost'])
    def apply(self, Y, X, perplexity):
        '''t-SNE embedding cost

        Parameters
        ----------

        X: `tensor.matrix`
            Data points in the orignal space

        Y: `tensor.matrix`
            Embedding points in the mapping space

        perplexity: float
            Perplexity measure to control the value of kernel size in the
            original space. Intuitevely, this is the number of neighbors
            each data point should have.

        alpha: float
            Degrees of freedom in the Student-t distribution

        References
        ----------
        .. [1] https://github.com/breze-no-salt/breze/blob/master/breze/learn/tsne.py
        '''
        q = self._get_probabilities_q(Y)
        p = self._get_probabilities_p(X, perplexity)

        # t-distributed stochastic neighbourhood embedding loss.
        loss = (p * tensor.log(p / q)).sum(axis=-1)
        return loss

    def _get_probabilities_q(self, X):
        dists = distance_matrix(X)
        top = zero_diagonal(1 / (1 + dists))
        bottom = top.sum(axis=0)
        q = top / bottom
        q /= q.sum()
        q = tensor.maximum(q, 1e-12)
        return q

    def _get_probabilities_p(self, X, perplexity=30):
        dists = distance_matrix(X, norm=lambda x, axis: l2(x, axis=axis)**2)
        sigma = dists.sort(axis=1)[:, -perplexity] / 3
        top = tensor.exp(-dists/(2*sigma[:, None]**2))
        top = zero_diagonal(top)
        bottom = top.sum(axis=0)
        p = top / bottom
        p = p + p.T
        p /= p.sum()
        p = tensor.maximum(p, 1e-12)
        return p
