import inspect

import jax.numpy as np
from jax.experimental.optimizers import nesterov, make_schedule
from jax import lax, value_and_grad

from jxf.util import sum_tuples, format_dataset, Verbosity, convex_combination


class ConjugatePrior:
    r"""Interface for a conjugate prior distribution.

    If :math:`\eta` is a random variable that parameterizes
    an exponential family distribution,

    ..math:
        p(x \mid \eta) = h(x) \exp\{t(x)^\top \eta - A(\eta)}

    then a conjugate prior on :math:`\eta` is of the form,

    ..math:
        p(\eta) = f(\xi, \nu) \exp \{\xi^\top \eta - \nu A(\eta)}

    where :math:`\xi` are pseudo-observations and :math:`\nu` are
    prior pseudo-counts, These play the same role as the sufficient
    statistics :math:`t(x)` and the number of datapoints, respectively.

    The constructor should default to an uninformative and generally
    improper prior with :math:`\xi` and :math:`\nu` both set to zero.
    In this case, MAP estimation reduces to maximum likelihood estimation.
    """
    @classmethod
    def from_stats(cls, stats, counts):
        """
        Construct an instance of the prior distribution given
        sufficient statistics and counts.
        """
        raise NotImplementedError

    @property
    def pseudo_obs(self):
        """Return the pseudo observations under this prior.
        These should match up with the sufficient statistics of
        the conjugate distribution.
        """
        raise NotImplementedError

    @property
    def pseudo_counts(self):
        """Return the pseudo observations under this prior."""
        raise NotImplementedError

    @property
    def log_normalizer(self):
        """Compute the log normalizer of the prior distribution.
        This is useful for computing marginal likelihoods in conjugate
        models.
        """
        raise NotImplementedError


class ExponentialFamilyDistribution:
    r"""An interface for exponential family distributions
    with the necessary functionality for MAP estimation.

    ..math:
        p(x) = h(x) \exp\{t(x)^\top \eta - A(\eta)}

    where

    :math:`h(x)` is the base measure
    :math:`t(x)` are sufficient statistics
    :math:`\eta` are natural parameters
    :math:`A(\eta)` is the log normalizer

    """
    @classmethod
    def from_params(cls, params, **kwargs):
        """Create an instance parameters of the distribution
        with given parameters (e.g. the mode of a posterior distribution
        on those parameters). This function might have to do some conversion,
        e.g. from variances to scales.
        """
        raise NotImplementedError

    @staticmethod
    def log_normalizer(params, **kwargs):
        """
        Return the log normalizer of the distribution.
        """
        raise NotImplementedError

    @staticmethod
    def sufficient_statistics(data, **kwargs):
        """
        Return the sufficient statistics for each datapoint in an array,
        This function should assume the leading dimensions give the batch
        size.
        """
        raise NotImplementedError

    @classmethod
    def fit_with_stats(cls,
                       sufficient_statistics,
                       num_datapoints,
                       prior=None,
                       **kwargs):
        """Compute the maximum a posteriori (MAP) estimate of the distribution
        parameters, given the sufficient statistics of the data and the number
        of datapoints.
        """
        # Compute the posterior distribution given sufficient statistics
        posterior_stats = sufficient_statistics
        posterior_counts = num_datapoints

        # Add prior stats if given
        if prior is not None:
            posterior_stats = sum_tuples(prior.pseudo_obs, posterior_stats)
            posterior_counts += prior.pseudo_counts

        # Construct the posterior
        posterior_class = get_expfam(cls)
        posterior = posterior_class.from_stats(posterior_stats, posterior_counts, **kwargs)

        # Return an instance of this distribution using the posterior mode parameters
        return cls.from_params(posterior.mode(), **kwargs)

    @classmethod
    @format_dataset
    def fit(cls, dataset, weights=None, prior=None, **kwargs):
        """Compute the maximum a posteriori (MAP) estimate of the distribution
        parameters.  For uninformative priors, this reduces to the maximum
        likelihood estimate.
        """
        # Compute the sufficient statistics and the number of datapoints
        suff_stats = None
        num_datapoints = 0
        for data_dict, these_weights in zip(dataset, weights):
            these_stats = cls.sufficient_statistics(**data_dict, **kwargs)

            # weight the statistics if weights are given
            if these_weights is not None:
                these_stats = tuple(np.tensordot(these_weights, s, axes=(0, 0))
                                    for s in these_stats)
            else:
                these_stats = tuple(s.sum(axis=0) for s in these_stats)

            # add to our accumulated statistics
            suff_stats = sum_tuples(suff_stats, these_stats)

            # update the number of datapoints
            num_datapoints += these_weights.sum()

        return cls.fit_with_stats(suff_stats, num_datapoints, prior=prior, **kwargs)

    @classmethod
    def proximal_optimizer(cls,
                           prior=None,
                           step_size=0.75,
                           **kwargs):
        """Return an optimizer triplet, like jax.experimental.optimizers,
        to perform proximal gradient ascent on the likelihood with a penalty
        on the KL divergence between distributions from one iteration to the
        next. This boils down to taking a convex combination of sufficient
        statistics from this data and those that have been accumulated from
        past data.

        Returns:

            initial_state    :: dictionary of optimizer state
                                (sufficient statistics and number of datapoints)
            update           :: minibatch, itr, state -> state
            get_distribution :: state -> Distribution object
        """
        initial_state = dict(suff_stats=None, num_datapoints=0.0)
        schedule = make_schedule(step_size)

        @format_dataset
        def update(itr,
                   dataset,
                   state,
                   weights=None,
                   suff_stats=None,
                   num_datapoints=0.0,
                   scale_factor=1.0):

            # Compute the sufficient statistics and the number of datapoints
            if suff_stats is None:
                num_datapoints = 0.0
                for data_dict, these_weights in zip(dataset, weights):
                    these_stats = cls.sufficient_statistics(**data_dict, **kwargs)

                    # weight the statistics if weights are given
                    if these_weights is not None:
                        these_stats = tuple(np.tensordot(these_weights, s, axes=(0, 0))
                                            for s in these_stats)
                    else:
                        these_stats = tuple(s.sum(axis=0) for s in these_stats)

                    # add to our accumulated statistics
                    suff_stats = sum_tuples(suff_stats, these_stats)

                    # update the number of datapoints
                    num_datapoints += these_weights.sum()
            else:
                # assume suff_stats and num_datapoints are given
                pass

            # Scale the sufficient statistics by the given scale factor.
            # This is as if the sufficient statistics were accumulated
            # from the entire dataset rather than a batch.
            suff_stats = tuple(scale_factor * ss for ss in suff_stats)
            num_datapoints = scale_factor * num_datapoints

            # Take a convex combination of sufficient statistics from
            # this batch and those accumulated thus far.
            if state["suff_stats"] is not None:
                state["suff_stats"] = convex_combination(
                    state["suff_stats"], suff_stats, schedule(itr))

                state["num_datapoints"] = convex_combination(
                    state["num_datapoints"], num_datapoints, schedule(itr))
            else:
                state = dict(suff_stats=suff_stats, num_datapoints=num_datapoints)

            return state

        def get_distribution(state):
            # Update parameters with the average stats
            return cls.fit_with_stats(state["suff_stats"],
                                      state["num_datapoints"],
                                      prior=prior,
                                      **kwargs)

        return initial_state, update, get_distribution


class CompoundDistribution:
    """Interface for compound distributions like the Student's t
    distribution and the negative binomial distribution.
    """
    @classmethod
    def from_params(cls, nonconjugate_params, conjugate_params, **kwargs):
        raise NotImplementedError

    @staticmethod
    def nonconj_params_to_unconstrained(nonconjugate_params, **kwargs):
        raise NotImplementedError

    @staticmethod
    def nonconj_params_from_unconstrained(value, **kwargs):
        raise NotImplementedError

    @staticmethod
    def initialize_params(dataset, weights, **kwargs):
        r"""Return intial values of the nonconjugate and conjugate
        parameters.  Each return value should be a tuple, even if
        there's only one parameter.
        """
        raise NotImplementedError

    @staticmethod
    def conditional_expectations(nonconjugate_params,
                                 conjugate_params,
                                 data,
                                 **kwargs):
        r"""Compute expectations under the conditional distribution
        over the auxiliary variables.  In the Student's t, for example,
        the auxiliary variables are the per-datapoint precision, :math:`\tau`,
        and the necessary expectations are :math:`\mathbb{E}[\tau]` and
        :math:`\mathbb{E}[\log \tau]`
        """
        raise NotImplementedError

    @staticmethod
    def expected_sufficient_statistics(nonconjugate_params,
                                       expectations,
                                       data,
                                       **kwargs):
        """Compute expected sufficient statistics necessary for a conjugate
        update of some parameters.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @classmethod
    @format_dataset
    def fit(cls, dataset,
            weights=None,
            prior=None,
            num_iters=100,
            tol=1e-4,
            nesterov_step_size=1e-1,
            nesterov_mass=0.9,
            nesterov_threshold=1e-8,
            nesterov_max_iters=100,
            verbosity=Verbosity.QUIET,
            **kwargs):
        """Fit a compound distribution use EM.
        """
        def e_step(nonconjugate_params, conjugate_params):
            # E step: compute conditional expectations
            return [cls.conditional_expectations(nonconjugate_params,
                                                 conjugate_params,
                                                 **data_dict,
                                                 **kwargs)
                    for data_dict in dataset]

        def conjugate_m_step(expectations, nonconjugate_params):
            # Compute expected sufficient statistics
            suff_stats = None
            num_datapoints = 0
            for expects, data_dict, these_weights in zip(expectations, dataset, weights):
                these_stats = cls.expected_sufficient_statistics(nonconjugate_params,
                                                                 expectations=expects,
                                                                 **data_dict,
                                                                 **kwargs)

                # weight the statistics if weights are given
                these_stats = tuple(np.tensordot(these_weights, s, axes=(0, 0))
                                    for s in these_stats)

                # add to our accumulated statistics
                suff_stats = sum_tuples(suff_stats, these_stats)

                # update the number of datapoints
                num_datapoints += these_weights.sum()

            # Find the optimal parameters for the conjugate part of the compound distribution
            posterior_stats = suff_stats
            posterior_counts = num_datapoints
            if prior is not None:
                posterior_stats = sum_tuples(prior.pseudo_obs, posterior_stats)
                posterior_counts += prior.pseudo_counts

            # Compute the posterior distribution
            posterior_class = get_compound(cls)
            posterior = posterior_class.from_stats(posterior_stats, posterior_counts, **kwargs)
            return posterior.mode()

        def nonconjugate_m_step(expectations, nonconjugate_params, conjugate_params):
            # M step: optimize the non-conjugate parameters via gradient methods.
            def objective(params):
                nonconjugate_params = cls.nonconj_params_from_unconstrained(params, **kwargs)
                lp = 0
                num_datapoints = 0
                for expects, data_dict, these_weights in zip(expectations, dataset, weights):
                    _lp = cls.expected_log_prob(nonconjugate_params,
                                                conjugate_params,
                                                expectations=expects,
                                                **data_dict,
                                                **kwargs)
                    lp += np.sum(these_weights * _lp)
                    num_datapoints += np.sum(these_weights)
                return -lp / num_datapoints

            # Optimize with Nesterov's accelerated gradient
            opt_init, opt_update, get_params = nesterov(nesterov_step_size, nesterov_mass)
            def check_convergence(state):
                itr, _, (prev_val, curr_val) = state
                return (abs(curr_val - prev_val) > nesterov_threshold) * (itr < nesterov_max_iters)

            def step(state):
                itr, opt_state, (_, prev_val) = state
                curr_val, grads = value_and_grad(objective)(get_params(opt_state))
                opt_state = opt_update(itr, grads, opt_state)
                return (itr + 1, opt_state, (prev_val, curr_val))

            # Initialize and run the optimizer
            init_params = cls.nonconj_params_to_unconstrained(nonconjugate_params, **kwargs)
            init_state = (0, opt_init(init_params), (np.inf, objective(init_params)))
            final_state = lax.while_loop(check_convergence, step, init_state)

            # Unpack the final state
            itr_count, params, lp = final_state[0], get_params(final_state[1]), -1 * final_state[2][1]
            if verbosity >= Verbosity.LOUD:
                print("Nesterov converged in ", itr_count, "iterations")

            return cls.nonconj_params_from_unconstrained(params, **kwargs), lp

        # Optimize the parameters with EM
        log_probs = []
        converged = False
        nonconjugate_params, conjugate_params = cls.initialize_params(dataset, weights, **kwargs)
        while not converged and len(log_probs) < num_iters:
            expectations = e_step(nonconjugate_params,
                                  conjugate_params)
            conjugate_params = conjugate_m_step(expectations,
                                                nonconjugate_params)
            nonconjugate_params, lp = nonconjugate_m_step(expectations,
                                                          nonconjugate_params,
                                                          conjugate_params)
            log_probs.append(lp)

            if len(log_probs) >= 2 and abs(log_probs[-1] - log_probs[-2]) < tol:
                if verbosity >= verbosity.LOUD:
                    print("log prob converged in ", len(log_probs), "iterations")
                converged = True

        return cls.from_params(nonconjugate_params, conjugate_params, **kwargs), np.array(log_probs)

# Initialize global dictionary to link exponential family distributions and
# their conjugate priors
__EXPFAM_DISTRIBUTIONS = dict()

def register_expfam(distribution, prior):
    __EXPFAM_DISTRIBUTIONS[distribution] = prior


def get_expfam(distribution):
    if inspect.isclass(distribution):
        return __EXPFAM_DISTRIBUTIONS[distribution]
    else:
        return __EXPFAM_DISTRIBUTIONS[type(distribution)]


# Initialize global dictionary to link compound distributions and
# their conditionally conjugate priors
__COMPOUND_DISTRIBUTIONS = dict()

def register_compound(distribution, prior):
    __COMPOUND_DISTRIBUTIONS[distribution] = prior


def get_compound(distribution):
    if inspect.isclass(distribution):
        return __COMPOUND_DISTRIBUTIONS[distribution]
    else:
        return __COMPOUND_DISTRIBUTIONS[type(distribution)]
