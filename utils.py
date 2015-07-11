import numpy as np
import theano

from sklearn.metrics.pairwise import pairwise_distances
from theano import tensor
from blocks.filter import VariableFilter, get_brick
from blocks.roles import OUTPUT, PARAMETER, add_role
from blocks.utils import shared_floatx
# from blocks.graph import apply_batch_normalization
floatX = theano.config.floatX


# def batch_normalize(mlp, cg):
#     variables = VariableFilter(bricks=mlp,
#                                roles=[OUTPUT])(cg.variables)
#     gammas = [shared_floatx(np.ones(get_brick(var).output_dim),
#                             name=var.name + '_gamma')
#               for var in variables]
#     for gamma in gammas:
#         add_role(gamma, PARAMETER)
#     betas = [shared_floatx(np.zeros(get_brick(var).output_dim),
#                            name=var.name + '_beta')
#              for var in variables]
#     for beta in betas:
#         add_role(beta, PARAMETER)
#     new_cg = apply_batch_normalization(cg, variables, gammas, betas, epsilon=1e-5)
#     return new_cg

def diff_abs(z):
    return tensor.sqrt(tensor.sqr(z)+1e-6)


def sparse_filtering_ff(z):
    l1 = tensor.sqrt(z**2 + 1e-8)
    rnorm = diff_abs(l1).sum(axis=1)
    l1row = l1 / rnorm[:, None, :]
    cnorm = diff_abs(l1row).sum(axis=2)
    l1col = l1row / cnorm[:, :, None]
    return l1col


def l2_norm_cost(mlp, cg, lbd):
    W = VariableFilter(bricks=mlp.linear_transformations, roles=[PARAMETER])(cg)
    cost = 0
    for w in W:
        cost = cost + lbd * tensor.sqr(w).sum()
    return cost


def _add_noise(n_steps, batch_size, dim, rfunc=np.random.normal):
    def func(data):
        noise = rfunc(0, 1, size=(n_steps, batch_size, dim)).astype(floatX)
        return (noise,)
    return func


def pairwise_diff(X, Y=None):
    '''Given two arrays with samples in the row, compute the pairwise
    differences.
    Parameters
    ----------

    X : Theano variable
        Has shape ``(n, d)``. Contains one item per first dimension.
    Y : Theano variable, optional [default: None]
        Has shape ``(m, d)``.  If not given, defaults to ``X``.

    Returns
    -------

    res : Theano variable
        Has shape ``(n, d, m)``.
    '''
    Y = X if Y is None else Y
    # diffs = X.T.dimshuffle(1, 0, 'x') - Y.T.dimshuffle('x', 0, 1)
    diffs = X[:, None, :] - X
    return diffs


def l2(X, axis=None):
    return tensor.sqrt((X**2).sum(axis=axis) + 1e-8)


def distance_matrix(X, Y=None, norm=l2):
    '''Return an expression containing the distances given the norm of up to two
    arrays containing samples.

    Parameters
    ----------

    X : Theano variable
        Has shape ``(n, d)``. Contains one item per first dimension.
    Y : Theano variable, optional [default: None]
        Has shape ``(m, d)``.  If not given, defaults to ``X``.
    norm : string or callable
        Either a string pointing at a function in ``breze.arch.component.norm``
        or a function that has the same signature as these.
    Returns
    -------

    res : Theano variable
        Has shape ``(n, m)``.

    References
    ----------
    .. [1] https://github.com/breze-no-salt/breze/blob/master/breze/learn/tsne.py
    '''
    diff = pairwise_diff(X, Y)
    dist = norm(diff, axis=-1)
    return dist


def zero_diagonal(X):
    '''Given a square matrix ``X``, return a theano variable with the diagonal
    of ``X`` set to zero.

    Parameters
    ----------

    X : theano 2d tensor

    Returns
    -------

    Y : theano 2d tensor

    References
    ----------
    .. [1] https://github.com/breze-no-salt/breze/blob/master/breze/learn/tsne.py
    '''
    thisid = tensor.identity_like(X)
    return (X - thisid * X)


def neighbour_probabilities(X, target_pplx):
    '''Return a square matrix containing probabilities that points given by `X`
    in the data are neighbours.
    Parameters
    ----------
    X : array_like
        2D array containing data points in rows of shape ``(n, d)``.
    target_pplx : float
        Desired perplexity.
    Returns
    -------
    P : array_like
        2D array of shape ``(n, n)`` containing probability that point ``i`` is
        neighbour of point j in ``P[i. j]``.
    '''
    N = X.shape[0]

    # Calculate the distances.
    dists = pairwise_distances(X, metric='euclidean', squared=True)

    # Parametrize in the log domain for positive standard deviations.
    precisions = np.ones(X.shape[0]).astype(theano.config.floatX)

    # Do a binary search for good logstds leading to the desired perplexities.
    minimums = np.empty(X.shape[0]).astype(theano.config.floatX)
    minimums[...] = -np.inf
    maximums = np.empty(X.shape[0]).astype(theano.config.floatX)
    maximums[...] = np.inf

    target_entropy = np.log(target_pplx)
    for i in range(50):
        # Calculate perplexities.
        inpt_top = np.exp(-dists * precisions)
        inpt_top[range(N), range(N)] = 0
        inpt_bottom = inpt_top.sum(axis=0)

        p_inpt_nb_cond = inpt_top / inpt_bottom
        # If we don't add a small term, the logarithm will make NaNs.
        p_inpt_nb_cond = np.maximum(1E-12, p_inpt_nb_cond)
        entropy = -(p_inpt_nb_cond * np.log(p_inpt_nb_cond)).sum(axis=0)

        diff = entropy - target_entropy
        for j in range(N):
            if abs(diff[j]) < 1e-5:
                continue
            elif diff[j] > 0:
                if maximums[j] == -np.inf or maximums[j] == np.inf:
                    precisions[j] *= 2
                else:
                    precisions[j] = (precisions[j] + maximums[j]) / 2
                minimums[j] = precisions[j]
            else:
                if minimums[j] == -np.inf or minimums[j] == np.inf:
                    precisions[j] /= 2
                else:
                    precisions[j] = (precisions[j] + minimums[j]) / 2
                maximums[j] = precisions[j]

        # Calculcate p matrix once more and return it.
        inpt_top = np.exp(-dists * precisions)
        inpt_top[range(N), range(N)] = 0
        inpt_bottom = inpt_top.sum(axis=0)
        # Add a small constant to the denominator against numerical issues.
        p_inpt_nb_cond = inpt_top / (inpt_bottom + 1e-4)

        # Symmetrize.
        p_ji = (p_inpt_nb_cond + p_inpt_nb_cond.T)

        # We don't normalize correctly here. But that does not matter, since we
        # normalize q wrongly in the same way.
        p_ji /= p_ji.sum()
        p_ji = np.maximum(1E-12, p_ji)

        return p_ji
