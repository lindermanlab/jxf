import jax.numpy as np
import jax.scipy.special as spsp
import jax.random
from jax.nn import softplus
from jax.tree_util import register_pytree_node_class
import tensorflow_probability.substrates.jax.distributions as tfp_dists

from jxf.base import ExponentialFamilyDistribution, CompoundDistribution, \
    register_expfam, register_compound
from jxf.distributions import MatrixNormalInverseWishart


class _LinearRegressionBase:
    """Base class for linear regressions.
    """
    def __init__(self, weights):
        self.weights = weights

    @property
    def data_dimension(self):
        return self.weights.shape[-2]

    @property
    def covariate_dimension(self):
        return self.weights.shape[-1]

    def predict(self, covariates):
        return covariates @ self.weights.T


@register_pytree_node_class
class GaussianLinearRegression(_LinearRegressionBase, ExponentialFamilyDistribution):
    """Linear regression with Gaussian noise.
    """
    def __init__(self, weights, covariance_matrix):
        super(GaussianLinearRegression, self).__init__(weights)
        self.covariance_matrix = covariance_matrix

    def tree_flatten(self):
        return ((self.weights, self.covariance_matrix), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_params(cls, params):
        return cls(*params)

    @staticmethod
    def sufficient_statistics(data, covariates):
        return (np.einsum('...i,...j->...ij', covariates, covariates),
                np.einsum('...i,...j->...ij', data, covariates),
                np.einsum('...i,...j->...ij', data, data))

    def sample(self, covariates, seed, sample_shape=()):
        d = tfp_dists.MultivariateNormalFullCovariance(
            self.predict(covariates), self.covariance_matrix)
        return d.sample(sample_shape=sample_shape, seed=seed)

    def log_prob(self, data, covariates, **kwargs):
        d = tfp_dists.MultivariateNormalFullCovariance(
            self.predict(covariates), self.covariance_matrix)
        return d.log_prob(data)


class MultivariateTLinearRegression(_LinearRegressionBase, CompoundDistribution):
    """A linear regression model. The optimal weights are computed
    via the expected sufficient statistics of the data.
    """
    def __init__(self, weights, df, scale):
        super(MultivariateTLinearRegression, self).__init__(weights)
        self.df = df
        self.scale = scale

    def tree_flatten(self):
        return ((self.weights, self.df, self.scale), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def dimension(self):
        return self.weights.shape[-2:]

    def log_prob(self, data, covariates):
        df, scale = self.df, self.scale
        dim = self.weights.shape[-2]
        predictions = covariates @ self.weights.T

        # Quadratic term
        tmp = np.linalg.solve(scale, (data - predictions).T).T
        lp = - 0.5 * (df + dim) * np.log1p(np.sum(tmp**2, axis=1) / df)

        # Normalizer
        lp += spsp.gammaln(0.5 * (df + dim)) - spsp.gammaln(0.5 * df)
        lp += - 0.5 * dim * np.log(np.pi) - 0.5 * dim * np.log(df)
        scale_diag = np.reshape(scale, scale.shape[:-2] + (-1,))[..., ::dim + 1]
        lp += -np.sum(np.log(scale_diag), axis=-1).reshape(scale.shape[:-2])
        return lp

    @classmethod
    def from_params(cls, nonconjugate_params, conjugate_params):
        df, = nonconjugate_params
        weights, covariance_matrix = conjugate_params
        return cls(weights, df, np.linalg.cholesky(covariance_matrix))

    @staticmethod
    def nonconj_params_to_unconstrained(nonconjugate_params):
        return (np.log(np.exp(nonconjugate_params[0]) - 1),)  # inverse of softplus

    @staticmethod
    def nonconj_params_from_unconstrained(value):
        return (softplus(value[0]),)

    # @staticmethod
    # # @format_dataset
    # def initialize_params(dataset, weights, **kwargs):
    #     # Initialize based on the mean and covariance of the data
    #     # loc, cov, num_datapoints = 0, 0, 0
    #     # for data_dict, these_weights in zip(dataset, weights):
    #     #     data = data_dict["data"]
    #     #     loc += np.einsum('n,ni->i', these_weights, data)
    #     #     cov += np.einsum('n,ni,nj->ij', these_weights, data, data)
    #     #     num_datapoints += these_weights.sum()

    #     # loc = loc / num_datapoints
    #     # cov = (cov / num_datapoints - np.outer(loc, loc))
    #     # df = loc.shape[0] + 2
    #     return (df,), (loc, cov)

    @staticmethod
    def conditional_expectations(nonconjugate_params,
                                 conjugate_params,
                                 data,
                                 covariates,
                                 **kwargs):
        """Compute expectations under the conditional distribution
        over the auxiliary variables.`
        """
        df, = nonconjugate_params
        weights, covariance_matrix = conjugate_params
        predictions = covariates @ weights.T
        scale = np.linalg.cholesky(covariance_matrix)
        dim = scale.shape[-1]

        # The auxiliary precision is conditionally gamma distributed.
        alpha = 0.5 * (df + dim)
        tmp = np.linalg.solve(scale, (data - predictions).T).T
        beta = 0.5 * (df + np.sum(tmp**2, axis=1))

        # Compute gamma expectations
        E_tau = alpha / beta
        E_log_tau = spsp.digamma(alpha) - np.log(beta)
        return E_tau, E_log_tau

    @staticmethod
    def expected_sufficient_statistics(nonconjugate_params,
                                       expectations,
                                       data,
                                       covariates,
                                       **kwargs):
        """Given the precision, the data is conditionally Gaussian.
        """
        E_tau, _ = expectations
        return (np.einsum('...,...i,...j->...ij', E_tau, covariates, covariates),
                np.einsum('...,...i,...j->...ij', E_tau, data, covariates),
                np.einsum('...,...i,...j->...ij', E_tau, data, data))

    @staticmethod
    def expected_log_prob(nonconjugate_params,
                          conjugate_params,
                          expectations,
                          data,
                          covariates,
                          **kwargs):
        """Compute the expected log probability.  This function will be
        optimized with respect to the remaining, non-conjugate parameters
        of the distribution.
        """
        df, = nonconjugate_params
        weights, covariance_matrix = conjugate_params
        scale = np.linalg.cholesky(covariance_matrix)
        scale_diag = np.reshape(scale, scale.shape[:-2] + (-1,))[..., ::dim + 1]
        predictions = covariates @ weights.T

        E_tau, E_log_tau = expectations
        hdof = 0.5 * df
        # lp = -np.sum(np.log(np.diag(scale)))
        lp = -np.sum(np.log(scale_diag), axis=-1).reshape(scale.shape[:-2])
        lp += 0.5 * E_log_tau
        lp += hdof * np.log(hdof)
        lp -= spsp.gammaln(hdof)
        lp += (hdof - 1) * E_log_tau
        lp -= hdof * E_tau
        tmp = np.linalg.solve(scale, (data - predictions).T).T
        lp -= 0.5 * E_tau * np.sum(tmp**2, axis=1)

        # Optional regularization on the degrees of freedom parameter
        lp -= 1e-8 * hdof  # regularization (exponential prior)
        return lp


# Register the regressions with their conjugate priors
register_expfam(GaussianLinearRegression, MatrixNormalInverseWishart)
register_compound(MultivariateTLinearRegression, MatrixNormalInverseWishart)


# Helper functions for autoregressions
def make_next_autoregression_covariate(data,
                                       num_lags=1,
                                       covariates=None,
                                       fit_intercept=True):
    num_data, data_dim = data.shape
    all_covariates = []
    for lag in range(1, num_lags+1):
        if lag > num_data:
            all_covariates.append(np.zeros(data_dim))
        else:
            all_covariates.append(data[-lag])

    if covariates is not None:
        all_covariates.append(covariates)

    if fit_intercept:
        all_covariates.append(np.ones(1))

    return np.concatenate(all_covariates, axis=-1)


def preprocess_autoregression_data(data,
                                   num_lags=1,
                                   covariates=None,
                                   fit_intercept=True,
                                   **kwargs):
    """Prepare a dataset for fitting with an AR model.

    This takes in data and returns a dictionary with data and covariates.
    """
    # prepare the covariates
    all_covariates = []
    for lag in range(1, num_lags+1):
        # TODO: Try to avoid memory allocation
        all_covariates.append(
            np.row_stack([np.zeros((lag, data.shape[-1])), data[..., :-lag, :]]))
    if covariates is not None:
        all_covariates.append(covariates)
    if fit_intercept:
        all_covariates.append(np.ones(data.shape[:-1] + (1,)))
    all_covariates = np.concatenate(all_covariates, axis=-1)

    # extract only the valid slice of the data and covariates
    valid_data = data[num_lags:]
    valid_covariates = all_covariates[num_lags:]
    # valid_data = data
    # valid_covariates = all_covariates

    return dict(data=valid_data, covariates=valid_covariates, **kwargs)
