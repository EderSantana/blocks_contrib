import numpy as np

from theano import tensor
from blocks.filter import VariableFilter, get_brick
from blocks.roles import OUTPUT, PARAMETER, add_role
from blocks.utils import shared_floatx
from blocks.graph import apply_batch_normalization


def batch_normalize(mlp, cg):
    variables = VariableFilter(bricks=mlp,
                               roles=[OUTPUT])(cg.variables)
    gammas = [shared_floatx(np.ones(get_brick(var).output_dim),
                            name=var.name + '_gamma')
              for var in variables]
    for gamma in gammas:
        add_role(gamma, PARAMETER)
    betas = [shared_floatx(np.zeros(get_brick(var).output_dim),
                           name=var.name + '_beta')
             for var in variables]
    for beta in betas:
        add_role(beta, PARAMETER)
    new_cg = apply_batch_normalization(cg, variables, gammas, betas, epsilon=1e-5)
    return new_cg


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
