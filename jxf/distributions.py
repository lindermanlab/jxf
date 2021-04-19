import jax.numpy as np
import jax.scipy.special as spsp
import jax.scipy.stats as spst
import jax.random
from jax import lax
from jax.nn import softplus
from jax.ops import index, index_add
from jax.tree_util import register_pytree_node, register_pytree_node_class
import tensorflow_probability.substrates.jax.distributions as tfp_dists

from jxf.base import ExponentialFamilyDistribution, \
    ConjugatePrior, register_expfam, CompoundDistribution, register_compound
from jxf.util import one_hot


class Bernoulli(ExponentialFamilyDistribution, tfp_dists.Bernoulli):
    r"""
    The Bernoulli distribution has pmf

    ..math:
        p(x) = \exp\{ x \log p + (1 - x) \log (1 - p) \}
             = \exp\{ x \log(p / (1 - p)) + \log (1 - p) \}
             = \exp\{ x \eta - A(\eta) \}

    In natural exponential family form, the Bernoulli distribution has

    - sufficient statistic: :math:`t(x) = x`,
    - natural parameter: :math:`\eta = \log p / (1-p) = \sigma^{-1}(p)`,
    - log normalizer: :math:`A(\eta) = \log(1 + e^{\eta}) = -\log(1 - p)`.
    """
    @classmethod
    def from_params(cls, params):
        return cls(probs=params)

    @staticmethod
    def sufficient_statistics(data):
        return (data,)


class Beta(ConjugatePrior, tfp_dists.Beta):
    r"""
    The beta distribution has pdf

    ..math:
        p(p) = 1/B(a, b) \exp\{(a-1) \log p + (b-1) \log (1-p) \}
             = 1/B(a, b) \exp\{(a-1) \log (p/(1-p))  + (a+b-2) * N / N \log (1-p)\}
             = 1/B(a, b) \exp\{(a-1) \eta  - (a+b-2) / N * A(\eta\}

    where :math:`\eta = \log(p / (1-p))` and :math:`A(\eta) = N \log(1 + e^{\eta})`,
    as in the Bernoulli and binomial definitions.

    The pseudo-observations are :math:`\chi = a - 1` and the
    pseudo-counts are :math:`\nu = (a + b - 2) / N`.
    """
    def __init__(self, *args, total_count=1, **kwargs):
        # `total_count` specifies the number of counts in the conjugate
        # likelihood (Bernoulli if total_count=1 or Binomial for >= 1).
        super(Beta, self).__init__(*args, **kwargs)
        self.total_count = total_count

    @classmethod
    def from_stats(cls, stats, counts, total_count=1):
        concentration1 = stats[0] + 1
        concentration0 = counts * total_count - concentration1 + 2
        return cls(concentration1, concentration0)

    @property
    def pseudo_obs(self):
        return (self.concentration1 - 1,)

    @property
    def pseudo_counts(self):
        """Return the pseudo observations under this prior."""
        return (self.concentration1 + self.concentration0 - 2) / self.total_count


class Binomial(ExponentialFamilyDistribution, tfp_dists.Binomial):
    r"""
    The Bernoulli distribution has pmf

    ..math:
        p(x) = \exp\{ x \log p + (N - x) \log (1 - p) \}
             = \exp\{ x \log(p / (1 - p)) + N \log (1 - p) \}
             = \exp\{ x \eta - A(\eta) \}

    In natural exponential family form, the Bernoulli distribution has

    - sufficient statistic: :math:`t(x) = x`,
    - natural parameter: :math:`\eta = \log p / (1-p) = \sigma^{-1}(p)`,
    - log normalizer: :math:`A(\eta) = N \log(1 + e^{\eta}) = -N \log(1 - p)`.
    """
    @classmethod
    def from_params(cls, params, total_count):
        return cls(total_count=total_count, probs=params)

    @staticmethod
    def sufficient_statistics(data, total_count):
        return (data,)


class Categorical(ExponentialFamilyDistribution, tfp_dists.Categorical):
    r"""
    The Categorical distribution has pmf

    ..math:
        p(x) = \exp\{ \sum_{k=1}^{K-1} [I[x=k] \log p_k] + I[x=K] \log p_K \}

    Note that :math:`I[x=K] = 1 - \sum_{k=1}^K I[x=k]`
    and :math:`p_K = 1 - \sum_{k=1}^K p_k`.

    We can rewrite the categorical pmf as follows,

    ..math:
        p(x) = \exp\{ \sum_{k=1}^{K-1} [I[x=k] \log (p_k / (1 - \sum_{j=1}^{K-1} p_j))]
                      + \log (1 - \sum_{j=1}^{K-1} p_j) \}

             = \exp\{ t(x)^\top \eta - A(\eta) \}

    where

    - sufficient statistic: :math:`t(x) = (I[x=1], ..., I[x=K-1])`,
    - natural parameter: :math:`\eta = (\log p_1 / (1-p_K), ..., \log p_{K-1} / (1-p_K))`,
    - log normalizer: :math:`A(\eta) = -\log(1 + \sum_{k=1}^{K-1} e^{\eta_k})`.
    """
    @classmethod
    def from_params(cls, params):
        return cls(probs=params)

    @staticmethod
    def sufficient_statistics(data):
        num_classes = data.max() + 1
        return (data[..., None] == np.arange(num_classes - 1),)


class Dirichlet(ConjugatePrior, tfp_dists.Dirichlet):
    r"""
    The Dirichlet distribution has pdf

    ..math:
        p(\pi) = 1/B(a) \exp\{\sum_{k=1}^K (a_k-1) \log \pi_k \}
               = 1/B(a) \exp\{\sum_{k=1}^{K-1} [(a_k-1) \log (\pi_k/\pi_K)]
                              + (\sum_{k=1}^K a_k - K) \log (1-\sum_{k=1}^{K-1} \pi_k)\}
               = 1/B(a) \exp\{\sum_{k=1}^{K-1}(a_k-1) \eta_k
                              - (\sum_{k=1}^K a_k - K) * N / N * A(\eta\}

    where :math:`\eta = (\log p_1 / (1-p_K), ..., \log p_{K-1} / (1-p_K))`,
    and :math:`A(\eta) = - N \log(1 + \sum_{k=1}^{K-1} e^{\eta_k})`,
    as in the categorical distribution.

    The pseudo-observations are :math:`\chi = a - 1` and the
    pseudo-counts are :math:`\nu = (\sum_{k=1}^K a_k - K) / N`.
    """
    def __init__(self, *args, total_count=1, **kwargs):
        # `total_count` specifies the number of counts in the conjugate
        # likelihood (Categorical if total_count=1 or Multinomial for >= 1).
        super(Dirichlet, self).__init__(*args, **kwargs)
        self.total_count = total_count

    @classmethod
    def from_stats(cls, stats, counts, total_count=1):
        concentration = stats[0] + 1
        num_classes = concentration.shape[-1] + 1
        last_concentration = np.atleast_1d(counts * total_count - concentration.sum(axis=-1) + num_classes)
        return cls(np.concatenate([concentration, last_concentration], axis=-1))

    @property
    def pseudo_obs(self):
        return (self.concentration[...,:-1] - 1,)

    @property
    def pseudo_counts(self):
        """Return the pseudo observations under this prior."""
        return (np.sum(self.concentration, axis=-1) - self.concentration.shape[-1]) / self.total_count


class Gamma(ConjugatePrior, tfp_dists.Gamma):
    r"""
    The Gamma pdf is

    ..math:
        p(\lambda) = 1/Z(\alpha, \beta) \exp\{(\alpha - 1) \log \lambda - \beta \lambda \}
                   = 1/Z(\alpha, \beta) \exp\{\xi \eta - \nu A(\eta) \}

    where :math:`\eta = \log \lambda` is a natural parameter of a conjugate exponential
    family distribution, and :math:`A(\eta) = e^\eta = \lambda` is its log normalizer.
    The gamma prior contributes pseudo-observations :math:`\xi = \alpha - 1` and
    pseudo-counts :math:`\nu = \beta`.

    For example, the gamma is conjugate to a Poisson likelihood,

    ..math:
        p(x \mid \lambda) = 1/x! \lambda^x e^{-\lambda}
                          = 1/x! \exp \{x \log \lambda - \lambda \}
                          = 1/x! \exp \{x \eta - A(\eta) \}
    """
    @classmethod
    def from_stats(cls, stats, counts):
        alpha, = stats
        return cls(alpha + 1, counts)

    @property
    def pseudo_obs(self):
        return (self.concentration - 1,)

    @property
    def pseudo_counts(self):
        """Return the pseudo observations under this prior."""
        return self.rate


@register_pytree_node_class
class MatrixNormalInverseWishart(object):
    r"""A conjugate prior distribution for a linear regression model,

    ..math::
        y | x ~ N(Ax, Sigma)

    where `A \in \mathbb{R}^{n \times p}` are the regression weights
    and `\Sigma \in \mathbb{S}^{n \times n}` is the noise covariance.
    Expanding the linear regression model,

    ..math::

        \log p(y | x) =
            -1/2 \log |\Sigma|
            -1/2 Tr((y - Ax)^\top \Sigma^{-1} (y - Ax))
          = -1/2 \log |\Sigma|
            -1/2 Tr(x x^\top A^\top \Sigma^{-1} A)
               + Tr(x y^\top \Sigma^{-1} A)
            -1/2 Tr(y y^\top \Sigma^{-1})

    Its natural parameters are
    .. math::
        \eta_1 = -1/2 A^\top \Sigma^{-1} A
        \eta_2 = \Sigma^{-1} A
        \eta_3 = -1/2 \Sigma^{-1}

    and they correspond to the sufficient statistics,
    .. math::
        t(x)_1 = x x^\top,
        t(x)_2 = y x^\top,
        t(x)_3 = y y^\top,

    The matrix-normal inverse-Wishart (MNIW) is a conjugate prior,

    ..math::
        A | \Sigma \sim \mathrm{N}(\vec(A) | \vec(M_0), \Sigma \kron V_0)
            \Sigma \sim \mathrm{IW}(\Sigma | \Psi_0, \nu_0)

    The prior parameters are:

        `M_0`: the prior mean of `A`
        `V_0`: the prior covariance of the columns of `A`
        `Psi_0`: the prior scale matrix for the noise covariance `\Sigma`
        `\nu_0`: the prior degrees of freedom for the noise covariance

    In the special case where the covariates are always one, `x = 1`, and
    hence the matrices `A` and `M_0` are really just column vectors `a` and
    `\mu_0`, the MNIW reduces to a NIW prior,

    ..math::
        a \sim \mathrm{NIW}{\mu_0, 1/V_0, \nu_0, \Psi_0}

    (`\kappa_0` is a precision in the NIW prior, whereas `V_0` is a covariance.)

    The MNIW pdf is proportional to,

    ..math::
        \log p(A , \Sigma) =
            -p/2 \log |\Sigma|
            -1/2 Tr(V_0^{-1} A^\top \Sigma^{-1} A)
               + Tr( V_0^{-1} M_0^\top \Sigma^{-1} A)
            -1/2 Tr(M_0 V_0^{-1} M_0^\top \Sigma^{-1})
            -(\nu_0 + n + 1)/2 \log|\Sigma|
            -1/2 Tr(\Psi_0 \Sigma^{-1})
            + c.

    Collecting terms, the prior contributes the following pseudo-counts
    and pseudo-observations,

    ..math::
        n_1 = \nu_0 + n + p + 1
        s_1 = V_0^{-1}
        s_2 = M_0 V_0^{-1}
        s_3 = \Psi_0 + M_0 V_0^{-1} M_0^\top

    We default to an improper prior, with `n_1 = 0` and
     `s_i = 0` for `i=1..3`.
    """
    def __init__(self, M0, V0, nu0, Psi0):
        symmetrize = lambda X: 0.5 * (X + X.T)
        self.M0 = M0
        self.V0 = symmetrize(V0)
        self.nu0 = nu0
        self.Psi0 = symmetrize(Psi0)

    def tree_flatten(self):
        return ((self.M0, self.V0, self.nu0, self.Psi0), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_stats(cls, stats, counts):
        r"""Convert the statistics and counts back into NIW parameters.
        Recall,

        ..math::
            n_1 = \nu_0 + n + p + 1
            s_1 = V_0^{-1}
            s_2 = M_0 V_0^{-1}
            s_3 = \Psi_0 + M_0 V_0^{-1} M_0^\top
        """
        s_1, s_2, s_3 = stats
        out_dim, in_dim = s_2.shape[-2:]

        nu0 = counts - out_dim - in_dim - 1
        def _null_stats(operand):
            V0 = 1e16 * np.eye(in_dim)
            M0 = np.zeros_like(s_2)
            Psi0 = np.eye(out_dim)
            return V0, M0, Psi0

        def _stats(operand):
            # TODO: Use Cholesky factorization for these two steps
            V0 = np.linalg.inv(s_1 + 1e-16 * np.eye(in_dim))
            M0 = s_2 @ V0
            Psi0 = s_3 - M0 @ s_1 @ M0.T
            return V0, M0, Psi0

        V0, M0, Psi0 = lax.cond(np.allclose(s_1, 0), _null_stats, _stats, operand=None)
        return cls(M0, V0, nu0, Psi0)

    @property
    def pseudo_obs(self):
        V0iM0T = np.linalg.solve(self.V0, self.M0.T)
        return (np.linalg.inv(self.V0),
                V0iM0T.T,
                self.Psi0 + self.M0 @ V0iM0T)

    @property
    def pseudo_counts(self):
        return self.nu0 + self.out_dim + self.in_dim + 1

    @property
    def in_dim(self):
        return self.M0.shape[-1]

    @property
    def out_dim(self):
        return self.M0.shape[-2]

    def log_prob(self, data, **kwargs):
        r"""Compute the prior log probability of LinearRegression weights
        and covariance matrix under this MNIW prior.  The IW pdf is provided
        in scipy.stats.  The matrix normal pdf is,

        ..math::
            \log p(A | M, \Sigma, V) =
                -1/2 Tr \left[ V^{-1} (A - M)^\top \Sigma^{-1} (A - M) \right]
                -np/2 \log (2\pi) -n/2 \log |V| -p/2 \log |\Sigma|

              = -1/2 Tr(B B^T) -np/2 \log (2\pi) -n/2 \log |V| -p/2 \log |\Sigma|

        where

        ..math::
            B = U^{-1/2} (A - M) (V^T)^{-1/2}
        """
        weights, covariance_matrix = data

        # Evaluate the matrix normal log pdf
        lp = 0

        # \log p(A | M_0, \Sigma, V_0)
        if np.all(np.isfinite(self.V0)):
            Vsqrt = np.linalg.cholesky(self.V0)
            Ssqrt = np.linalg.cholesky(covariance_matrix)
            B = np.linalg.solve(Ssqrt, np.linalg.solve(
                Vsqrt, (weights - self.M0).T).T)
            lp += -0.5 * np.sum(B**2)
            lp += -self.out_dim * np.sum(np.log(np.diag(Vsqrt)))
            lp += -0.5 * self.in_dim * self.out_dim * np.log(2 * np.pi)
            lp += -self.in_dim * np.sum(np.log(np.diag(Ssqrt)))

        # For comparison, compute the big multivariate normal log pdf explicitly
        # Note: we have to do the kron in the reverse order of what is given
        # on Wikipedia since ravel() is done in row-major ('C') order.
        # lp_test = scipy.stats.multivariate_normal.logpdf(
        #     np.ravel(weights), np.ravel(self.M0),
        #     np.kron(covariance_matrix, self.V0))
        # assert np.allclose(lp, lp_test)

        # \log p(\Sigma | \Psi0, \nu0)
        if self.nu0 >= self.out_dim and \
            np.all(np.linalg.eigvalsh(self.Psi0) > 0):
            # TODO: Use JAX versions of the logpdf's
            import scipy.stats
            lp += scipy.stats.invwishart.logpdf(
                covariance_matrix, self.nu0, self.Psi0)
        return lp

    def mode(self):
        r"""Solve for the mode. Recall,
        .. math::
            p(A, \Sigma) \propto
                \mathrm{N}(\vec(A) | \vec(M_0), \Sigma \kron V_0) \times
                \mathrm{IW}(\Sigma | \nu_0, \Psi_0)

        The optimal mean is :math:`A^* = M_0`. Substituting this in,
        .. math::
            p(A^*, \Sigma) \propto IW(\Sigma | \nu_0 + p, \Psi_0)

        and the mode of this inverse Wishart distribution is at
        .. math::
            \Sigma^* = \Psi_0 / (\nu_0 + p + n + 1)
        """
        A = self.M0
        Sigma = self.Psi0 / (self.nu0 + self.in_dim + self.out_dim + 1)
        Sigma += 1e-4 * np.eye(self.out_dim)
        return A, Sigma


class Multinomial(ExponentialFamilyDistribution, tfp_dists.Multinomial):
    r"""
    The Multinomial distribution has pmf

    ..math:
        p(x \mid N, p) \propto \exp\{ \sum_{k=1}^{K-1} [N_k \log p_k] + N_K \log p_K \}

    Note that :math:`N_k = N - \sum_{k=1}^K N_k`
    and :math:`p_K = 1 - \sum_{k=1}^K p_k`.

    We can rewrite the categorical pmf as follows,

    ..math:
        p(x) \propto \exp\{ \sum_{k=1}^{K-1} [N_k \log (p_k / (1 - \sum_{j=1}^{K-1} p_j))]
                      + N \log (1 - \sum_{j=1}^{K-1} p_j) \}

             = \exp\{ t(x)^\top \eta - A(\eta) \}

    where

    - sufficient statistic: :math:`t(x) = (I[x=1], ..., I[x=K-1])`,
    - natural parameter: :math:`\eta = (\log p_1 / (1-p_K), ..., \log p_{K-1} / (1-p_K))`,
    - log normalizer: :math:`A(\eta) = -N \log(1 + \sum_{k=1}^{K-1} e^{\eta_k})`.
    """
    @classmethod
    def from_params(cls, params, total_count):
        return cls(total_count=total_count, probs=params)

    @staticmethod
    def sufficient_statistics(data, total_count):
        return (data[..., :-1],)


class MultivariateNormalFullCovariance(ExponentialFamilyDistribution,
                                       tfp_dists.MultivariateNormalFullCovariance):
    """The multivariate normal distribution is both an exponential family
    distribution as well as a conjugate prior (for the mean of a multivariate
    normal distribution)."""

    @classmethod
    def from_params(cls, params, **kwargs):
        return cls(*params, **kwargs)

    @staticmethod
    def sufficient_statistics(data, **kwargs):
        if "covariance_matrix" in kwargs:
            Sigma = kwargs["covariance_matrix"]
            transpose = lambda x: np.swapaxes(x, -1, -2)
            return (transpose(np.linalg.solve(Sigma, transpose(data))),)
        else:
            return (np.ones(data.shape[:-1]),
                    data,
                    np.einsum('...i,...j->...ij', data, data))


class MultivariateNormalTriL(ExponentialFamilyDistribution,
                             tfp_dists.MultivariateNormalTriL):
    @classmethod
    def from_params(cls, params):
        loc, covariance = params
        return cls(loc, np.linalg.cholesky(covariance))

    @staticmethod
    def sufficient_statistics(data):
        return (np.ones(data.shape[:-1]),
                data,
                np.einsum('...i,...j->...ij', data, data))


@register_pytree_node_class
class MultivariateStudentTTril(CompoundDistribution):
    """A multivariate Student's T distribution with the CompoundDistribution
    interface for fitting with EM.
    """
    def __init__(self, df, loc, scale):
        self.df = df
        self.loc = loc
        self.scale = scale

    def tree_flatten(self):
        return ((self.df, self.loc, self.scale), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def dimension(self):
        return self.loc.shape[0]

    def log_prob(self, data, **kwargs):
        loc, scale, df, dim = \
            self.loc, self.scale, self.df, self.dimension
        assert data.ndim == 2 and data.shape[1] == dim

        # Quadratic term
        tmp = np.linalg.solve(scale, (data - loc).T).T
        lp = - 0.5 * (df + dim) * np.log1p(np.sum(tmp**2, axis=1) / df)

        # Normalizer
        lp += spsp.gammaln(0.5 * (df + dim)) - spsp.gammaln(0.5 * df)
        lp += - 0.5 * dim * np.log(np.pi) - 0.5 * dim * np.log(df)
        # L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]
        lp += -np.sum(np.log(np.diag(scale)))
        return lp

    @classmethod
    def from_params(cls, nonconjugate_params, conjugate_params):
        df, = nonconjugate_params
        loc, covariance_matrix = conjugate_params
        return cls(df, loc, np.linalg.cholesky(covariance_matrix))

    @staticmethod
    def nonconj_params_to_unconstrained(nonconjugate_params):
        return (np.log(np.exp(nonconjugate_params[0]) - 1),)  # inverse of softplus

    @staticmethod
    def nonconj_params_from_unconstrained(value):
        return (softplus(value[0]),)

    @staticmethod
    # @format_dataset
    def initialize_params(dataset, weights, **kwargs):
        # Initialize based on the mean and covariance of the data
        loc, cov, num_datapoints = 0, 0, 0
        for data_dict, these_weights in zip(dataset, weights):
            data = data_dict["data"]
            loc += np.einsum('n,ni->i', these_weights, data)
            cov += np.einsum('n,ni,nj->ij', these_weights, data, data)
            num_datapoints += these_weights.sum()

        loc = loc / num_datapoints
        cov = (cov / num_datapoints - np.outer(loc, loc))
        df = loc.shape[0] + 2
        return (df,), (loc, cov)

    @staticmethod
    def conditional_expectations(nonconjugate_params,
                                 conjugate_params,
                                 data,
                                 **kwargs):
        """Compute expectations under the conditional distribution
        over the auxiliary variables.`
        """
        df, = nonconjugate_params
        loc, covariance_matrix = conjugate_params
        scale = np.linalg.cholesky(covariance_matrix)
        dim = loc.shape[-1]

        # The auxiliary precision is conditionally gamma distributed.
        alpha = 0.5 * (df + dim)
        tmp = np.linalg.solve(scale, (data - loc).T).T
        beta = 0.5 * (df + np.sum(tmp**2, axis=1))

        # Compute gamma expectations
        E_tau = alpha / beta
        E_log_tau = spsp.digamma(alpha) - np.log(beta)
        return E_tau, E_log_tau

    @staticmethod
    def expected_sufficient_statistics(nonconjugate_params,
                                       expectations,
                                       data,
                                       **kwargs):
        """Given the precision, the data is conditionally Gaussian.
        """
        E_tau, _ = expectations
        return (E_tau,
                np.einsum('n,ni->ni', E_tau, data),
                np.einsum('n,ni,nj->nij', E_tau, data, data))

    @staticmethod
    def expected_log_prob(nonconjugate_params,
                          conjugate_params,
                          expectations,
                          data,
                          **kwargs):
        """Compute the expected log probability.  This function will be
        optimized with respect to the remaining, non-conjugate parameters
        of the distribution.
        """
        df, = nonconjugate_params
        loc, covariance_matrix = conjugate_params
        scale = np.linalg.cholesky(covariance_matrix)

        E_tau, E_log_tau = expectations
        hdof = 0.5 * df
        lp = -np.sum(np.log(np.diag(scale)))
        lp += 0.5 * E_log_tau
        lp += hdof * np.log(hdof)
        lp -= spsp.gammaln(hdof)
        lp += (hdof - 1) * E_log_tau
        lp -= hdof * E_tau
        # The quadratic term is in expectation zero at the optimal params,
        # and it doesn't depend on the dof, so these two lines could be dropped
        tmp = np.linalg.solve(scale, (data - loc).T).T
        lp -= 0.5 * E_tau * np.sum(tmp**2, axis=1)

        # Optional regularization on the degrees of freedom parameter
        lp -= 1e-8 * hdof  # regularization (exponential prior)
        return lp


class Normal(ExponentialFamilyDistribution, tfp_dists.Normal):
    @classmethod
    def from_params(cls, params):
        loc, variance = params
        return cls(loc, np.sqrt(variance))

    @staticmethod
    def sufficient_statistics(data):
        return (np.ones_like(data), data, data**2)


@register_pytree_node_class
class NormalInverseGamma(ConjugatePrior):
    r"""A conjugate prior distribution for the normal distribution.
    ..math::
    \mu | \sigma^2 \sim \mathrm{N}(\mu | \mu_0, \sigma^2 / \kappa_0)
          \sigma^2 \sim \mathrm{IGa}(\sigma^2 | \alpha_0, \beta_0)

    and the log pdf is proportional to,

    ..math::
    \log p(\mu, \sigma^2) =
        -(\alpha_0 + 1 + 0.5) \log \sigma^2
        + \kappa_0 * (-0.5 \mu^2 / \sigma^2)
        + \kappa_0 \mu_0 \mu / \sigma^2
        + (2 * \beta_0 + \kappa_0 \mu_0^2) * (-0.5 / \sigma^2)

    The natural parameters of the normal distribution are,
    .. math::
    \eta_1 = -0.5 \mu^2 / \sigma^2
    \eta_2 = \mu / \sigma^2,
    \eta_3 = -0.5 / \sigma^2,

    and they correspond to the sufficient statistics of the normal likelihood,
    .. math::
        t(x)_1 = 1
        t(x)_2 = x,
        t(x)_3 = x^2

    This looks a bit unconventional in that the first sufficient statistic
    is actually not a function of the data.  This parameterization makes
    sense when we write the normal distribution as a linear regression,

    ..math::
        y \mid x=1 \sim N(y \mid \mu x, \sigma^2)

    where :math:`x=1` is a fixed covariate.  In that case, the first sufficient
    statistic becomes :math:`t(x) = x^2 = 1`.  See the
    `MultivariateNormalLinearRegression` class for more detail.

    The NIG prior provides the following pseudo-observations and pseudo-counts:
    .. math:
        s_1 = \kappa_0
        s_2 = \kappa_0 \mu_0
        s_3 = 2 * \beta_0 + \kappa_0 \mu_0^2
        n = \alpha_0 + 1.5

    We default to an improper uniform prior with zero pseudo counts"""
    def __init__(self, mu0, kappa0, alpha0, beta0):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def tree_flatten(self):
        return ((self.mu0, self.kappa0, self.alpha0, self.beta0), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_stats(cls, stats, counts):
        r"""
        Convert the statistics and counts back into NIG parameters.
        .. math:
            s_1 = \kappa_0
            s_2 = \kappa_0 \mu_0
            s_3 = 2 * \beta_0 + \kappa_0 \mu_0^2
            n = 2 \alpha_0 + 3
        """
        s_1, s_2, s_3 = stats
        kappa0 = s_1
        mu0 = s_2 / kappa0 if kappa0 > 0 else np.zeros_like(s_1)
        beta0 = 0.5 * (s_3 - kappa0 * mu0**2)
        alpha0 = 0.5 * (counts - 3)
        return cls(mu0, kappa0, alpha0, beta0)

    @property
    def pseudo_obs(self):
        return (self.kappa0,
                self.kappa0 * self.mu0,
                2 * self.beta0 + self.kappa0 * self.mu0**2)

    @property
    def pseudo_counts(self):
        return 2 * self.alpha0 + 3

    def log_prob(self, value):
        mean, variance = value

        lp = 0
        if self.kappa0 > 0:
            lp += np.sum(spst.norm.logpdf(
                mean, self.mu0, np.sqrt(variance / self.kappa0)))

        if self.alpha0 > 0:
            # TODO: Use JAX versions of the logpdf's
            import scipy.stats
            lp += scipy.stats.invgamma.logpdf(
                variance, self.alpha0, scale=self.beta0)
        return lp

    def mode(self):
        r"""Solve for the mode. Recall,

        ..math::
            p(\mu, \sigma^2) \propto
                \mathrm{N}(\mu | \mu_0, \sigma^2 / \kappa_0) \times
                \mathrm{IGa}(\Sigma | \alpha_0, \beta_0)

        The optimal mean is :math:`\mu^* = \mu_0`. Substituting this in,
        ..math::
            p(\mu^*, \sigma^2) \propto IGa(\sigma^2 | \alpha_0 + 0.5, \beta_0)

        and the mode of this inverse gamma distribution is at
        ..math::
            (\sigma^2)* = \beta_0 / (\alpha_0 + 1.5)
        """
        return self.mu0, self.beta0 / (self.alpha0 + 1.5)


@register_pytree_node_class
class NormalInverseWishart(object):
    def __init__(self, mu0, kappa0, nu0, Psi0):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.nu0 = nu0
        self.Psi0 = Psi0

    def tree_flatten(self):
        return ((self.mu0, self.kappa0, self.nu0, self.Psi0), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_stats(cls, stats, counts):
        s_1, s_2, s_3 = stats
        dim = s_2.shape[-1]

        kappa0 = s_1
        mu0 = lax.cond(kappa0 > 0,
                    lambda x: s_2 / kappa0,
                    lambda x: np.zeros_like(s_2),
                    None)
        Psi0 = s_3 - kappa0 * np.einsum('...i,...j->...ij', mu0, mu0)
        nu0 = counts - dim - 2
        return cls(mu0, kappa0, nu0, Psi0)

    @property
    def pseudo_obs(self):
        return (self.kappa0,
                self.kappa0 * self.mu0,
                self.Psi0 + self.kappa0 * np.einsum('...i,...j->...ij', self.mu0, self.mu0))

    @property
    def pseudo_counts(self):
        return self.nu0 + self.dim + 1

    @property
    def dim(self):
        return self.mu0.shape[-1]

    def log_prob(self, data, **kwargs):
        """Compute the prior log probability of a MultivariateNormal
        distribution's parameters under this NIW prior.

        Note that the NIW prior is only properly specified in certain
        parameter regimes (otherwise the density does not normalize).
        Only compute the log prior if there is a density.
        """
        mean, covariance_matrix = data
        assert mean.shape[0] == self.dim

        lp = 0
        if self.kappa0 > 0:
            lp += np.sum(spst.multivariate_normal.logpdf(
                mean, self.mu0, covariance_matrix / self.kappa0))

        if self.nu0 >= self.dim:
            # TODO: Use JAX versions of the logpdf's
            import scipy.stats
            lp += scipy.stats.invwishart.logpdf(
                covariance_matrix, self.nu0, self.Psi0)
        return lp

    def mode(self):
        r"""Solve for the mode. Recall,

        .. math::
            p(\mu, \Sigma) \propto
                \mathrm{N}(\mu | \mu_0, \Sigma / \kappa_0) \times
                \mathrm{IW}(\Sigma | \nu_0, \Psi_0)

        The optimal mean is :math:`\mu^* = \mu_0`. Substituting this in,
        .. math::
            p(\mu^*, \Sigma) \propto IW(\Sigma | \nu_0 + 1, \Psi_0)

        and the mode of this inverse Wishart distribution is at
        .. math::
            \Sigma^* = \Psi_0 / (\nu_0 + d + 2)

        """
        return self.mu0, self.Psi0 / (self.nu0 + self.dim + 2)


class Poisson(ExponentialFamilyDistribution, tfp_dists.Poisson):
    @classmethod
    def from_params(cls, params):
        return cls(rate=params)

    @staticmethod
    def sufficient_statistics(data):
        return (data,)


class StudentT(CompoundDistribution, tfp_dists.StudentT):
    """A Student's T distribution with the CompoundDistribution interface
    for fitting with EM.
    """
    @classmethod
    def from_params(cls, nonconjugate_params, conjugate_params):
        df, = nonconjugate_params
        loc, variance = conjugate_params
        return cls(df, loc, np.sqrt(variance))

    @staticmethod
    def nonconj_params_to_unconstrained(nonconjugate_params):
        df, = nonconjugate_params
        return (np.log(np.exp(df) - 1),)  # inverse of softplus

    @staticmethod
    def nonconj_params_from_unconstrained(value):
        inv_sftpls_df, = value
        return (softplus(inv_sftpls_df),)

    @staticmethod
    # @format_dataset
    def initialize_params(dataset, weights, **kwargs):
        # Initialize based on the mean and covariance of the data
        loc, var, num_datapoints = 0, 0, 0
        for data_dict, these_weights in zip(dataset, weights):
            data = data_dict["data"]
            # loc += np.einsum('n,ni->i', these_weights, data)
            # var += np.einsum('n,ni->i', these_weights, data**2)
            loc += np.tensordot(these_weights, data, axes=(0, 0))
            var += np.tensordot(these_weights, data**2, axes=(0, 0))
            num_datapoints += these_weights.sum()

        loc = loc / num_datapoints
        var = (var / num_datapoints - loc**2)
        df = 3.0
        return (df,), (loc, var)

    @staticmethod
    def conditional_expectations(nonconjugate_params,
                                 conjugate_params,
                                 data,
                                 **kwargs):
        """Compute expectations under the conditional distribution
        over the auxiliary variables.`
        """
        df, = nonconjugate_params
        loc, variance = conjugate_params

        # The auxiliary precision \tau is conditionally gamma distributed.
        alpha = 0.5 * (df + 1)
        beta = 0.5 * (df + (data - loc)**2 / variance)

        # Compute gamma expectations
        E_tau = alpha / beta
        E_log_tau = spsp.digamma(alpha) - np.log(beta)
        return E_tau, E_log_tau

    @staticmethod
    def expected_sufficient_statistics(nonconjugate_params,
                                       expectations,
                                       data,
                                       **kwargs):
        """Given the precision, the data is conditionally Gaussian.
        """
        E_tau, _ = expectations
        return E_tau, E_tau * data, E_tau * data**2

    @staticmethod
    def expected_log_prob(nonconjugate_params,
                          conjugate_params,
                          expectations,
                          data,
                          **kwargs):
        """Compute the expected log probability.  This function will be
        optimized with respect to the remaining, non-conjugate parameters
        of the distribution.
        """
        df, = nonconjugate_params
        # loc, variance = conjugate_params
        # scale = np.sqrt(variance)

        E_tau, E_log_tau = expectations
        hdof = 0.5 * df
        return hdof * np.log(hdof) \
            - spsp.gammaln(hdof) + (hdof - 1) * E_log_tau \
            - hdof * E_tau

        # E_tau, E_log_tau = expectations
        # hdof = 0.5 * df
        # lp = -np.sum(np.log(np.diag(scale)))
        # lp += 0.5 * E_log_tau
        # lp += hdof * np.log(hdof)
        # lp -= spsp.gammaln(hdof)
        # lp += (hdof - 1) * E_log_tau
        # lp -= hdof * E_tau
        # # The quadratic term is in expectation zero at the optimal params,
        # # and it doesn't depend on the dof, so these two lines could be dropped
        # tmp = np.linalg.solve(scale, (data - loc).T).T
        # lp -= 0.5 * E_tau * np.sum(tmp**2, axis=1)

        # # Optional regularization on the degrees of freedom parameter
        # lp -= 1e-8 * hdof  # regularization (exponential prior)
        # return lp


# Register exponential family relationships
register_expfam(Bernoulli, Beta)
register_expfam(Binomial, Beta)
register_expfam(Categorical, Dirichlet)
register_expfam(Multinomial, Dirichlet)
register_expfam(MultivariateNormalFullCovariance, NormalInverseWishart)
register_expfam(MultivariateNormalTriL, NormalInverseWishart)
register_expfam(Normal, NormalInverseGamma)
register_expfam(Poisson, Gamma)

# Register compound distributions
register_compound(MultivariateStudentTTril, NormalInverseWishart)
register_compound(StudentT, NormalInverseGamma)
