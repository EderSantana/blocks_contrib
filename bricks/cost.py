import numpy as np

from theano import tensor

from blocks.bricks.base import application
from blocks.bricks.cost import Cost


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
