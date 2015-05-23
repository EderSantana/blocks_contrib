import numpy as np
import theano

from theano import tensor

from blocks.bricks.base import application
from blocks.bricks.cost import Cost

# from blocks_contrib.utils import distance_matrix, zero_diagonal, l2
# from _breze_misc import distance_matrix
floatX = theano.config.floatX
MACHINE_EPSILON = np.finfo(np.double).eps


def _zero_diagonal(X):
    '''zero out the main diagonal
    '''
    diag_matrix = tensor.indentity_like(X)
    return (X - diag_matrix * X)


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
    def apply(self, Y, P, alpha=1):
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
        .. [1] van der Maaten, L.J.P.; Hinton, G.E. (Nov 2008).
               "Visualizing High-Dimensional Data Using t-SNE". JMLR.
        .. [2] https://github.com/breze-no-salt/breze/blob/master/breze/learn/tsne.py

        '''
        Q = self._get_probabilities_q(Y, alpha)
        # p = self._get_probabilities_p(X, perplexity)

        # t-distributed stochastic neighbourhood embedding loss.
        loss = (P * tensor.log(P / Q)).sum()
        return loss

    def _get_probabilities_q(self, X, alpha):
        n = ((X[:, None, :] - X)**2).sum(axis=-1)
        n += 1.
        n /= alpha
        n **= (alpha + 1.0) / -2.0
        n = _zero_diagonal(n / (2.0 * tensor.sum(n)))
        Q = tensor.maximum(n, MACHINE_EPSILON)
        # dists = distance_matrix(X)
        return Q

    # def _get_probabilities_p(self, X, perplexity=30):
    #     dists = distance_matrix(X, norm=lambda x, axis: l2(x, axis=axis)**2)
    #     sigma = 1  # dists.sort(axis=1)[:, -perplexity]
    #     top = tensor.exp(-dists/(2*sigma**2))
    #     # top = zero_diagonal(top)
    #     bottom = top.sum(axis=0)
    #     p = top / bottom
    #     p = p + p.T
    #     p /= p.sum()
    #     p = tensor.maximum(p, 1e-12)
    #     return p
