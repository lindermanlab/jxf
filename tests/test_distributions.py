import time
import jax.numpy as np
import jax.random as jr
import jxf.distributions as dists
import jxf.regressions as regrs

def test_normal(shape=(10000,)):
    key = jr.PRNGKey(time.time_ns())
    data = 5 * jr.normal(key, shape=shape)
    norm = dists.Normal.fit(data)
    assert np.allclose(data.mean(), norm.loc)
    assert np.allclose(data.std(), norm.scale)

# Students T
def test_studentst(shape=(1000,), loc=0.0, scale=5.0, dof=5.0):
    key = jr.PRNGKey(time.time_ns())
    key1, key2 = jr.split(key, 2)
    zs = jr.normal(key1, shape=shape)
    alpha, beta = dof / 2.0, 2.0 / dof
    taus = jr.gamma(key2, alpha, shape=shape)  / beta
    data = loc + zs * scale / np.sqrt(taus)
    # true = dists.StudentT(dof, loc, scale)
    # data = true.sample(seed=key, sample_shape=shape)
    norm = dists.Normal.fit(data)
    stdt, lps = dists.StudentT.fit(data)
    assert np.all(np.diff(lps) > -1e-3)
    assert stdt.log_prob(data).mean() > norm.log_prob(data).mean()

# Multivariate normal
def test_mvn(shape=(1000, 5)):
    key = jr.PRNGKey(time.time_ns())
    data = 5 * jr.normal(key, shape=shape)
    mvn = dists.MultivariateNormalFullCovariance.fit(data)
    assert np.allclose(data.mean(axis=0), mvn.loc, atol=1e-6)
    assert np.allclose(np.cov(data, rowvar=False, bias=True), mvn.covariance(), atol=1e-6)

# Multivariate Student's t
# def test_mvt(shape=(1000,), dim=5, loc=0.0, scale=5.0, dof=5.0):
#     key = jr.PRNGKey(time.time_ns())
#     true = dists.MultivariateStudentTTril(dof, loc * np.zeros(dim), scale * np.eye(dim))
#     data = true.sample(key, sample_shape=shape)
#     mvn = dists.MultivariateNormalFullCovariance.fit(data)
#     mvt, lps = dists.MultivariateStudentTTril.fit(data)
#     # assert np.allclose(data.mean(axis=0), mvn.loc, atol=1e-6)
#     # assert np.allclose(np.cov(data, rowvar=False, bias=True), mvn.covariance(), atol=1e-6)
#     assert np.all(np.diff(lps) > -1e-3)
#     assert mvt.log_prob(data).mean() > mvn.log_prob(data).mean()

# Bernoulli
def test_bernoulli(shape=(1000,)):
    key = jr.PRNGKey(time.time_ns())
    beta = dists.Beta(1, 1)
    data = jr.bernoulli(key, 0.5, shape=shape)
    bern = dists.Bernoulli.fit(data, prior=beta)
    assert np.allclose(data.mean(), bern.probs_parameter(), atol=1e-6)

# Binomial
def test_binomial(shape=(1000,), total_count=10):
    key = jr.PRNGKey(time.time_ns())
    beta = dists.Beta(1, 1, total_count=total_count)
    data = jr.bernoulli(jr.PRNGKey(0), 0.5, shape=(100, total_count)).sum(axis=1)
    bino = dists.Binomial.fit(data, prior=beta, total_count=total_count)
    assert np.allclose((data / total_count).mean(), bino.probs, atol=1e-6)

# Categorical
def test_categorical(shape=(1000,), num_classes=5):
    key = jr.PRNGKey(time.time_ns())
    diri = dists.Dirichlet(np.ones(num_classes))
    data = jr.choice(key, num_classes, shape=shape)
    cate = dists.Categorical.fit(data, prior=diri,)
    assert np.allclose(np.bincount(data, minlength=5) / len(data),
                       cate.probs_parameter(), atol=1e-6)


# Multinomial
def test_multinomial(shape=(1000,), num_classes=5, total_count=10):
    key = jr.PRNGKey(time.time_ns())
    diri = dists.Dirichlet(np.ones(num_classes))
    inds = jr.choice(key, num_classes, shape=shape + (total_count,))
    inds = inds[..., None] == np.arange(num_classes)
    data = inds.sum(axis=(-2))
    mult = dists.Multinomial.fit(data, prior=diri, total_count=total_count)
    assert np.allclose(data.mean(axis=0) / total_count, mult.probs_parameter(), atol=1e-6)


# Poisson
def test_poisson(shape=(1000, 5)):
    key = jr.PRNGKey(time.time_ns())
    data = jr.poisson(key, 5.0, shape=shape)
    pois = dists.Poisson.fit(data)
    assert np.allclose(data.mean(axis=0), pois.rate, atol=1e-6)

# Linear regression
def test_linear_regression(in_dim=5, out_dim=2, shape=(1000,)):
    key = jr.PRNGKey(time.time_ns())
    key1, key2 = jr.split(key, 2)
    covariates = jr.normal(key1, shape + (in_dim,))
    covariates = np.column_stack([covariates, np.ones(shape)])
    data = jr.normal(key2, shape + (out_dim,))
    lr = regrs.GaussianLinearRegression.fit(dict(data=data, covariates=covariates))

    # compare to least squares fit.  note that the covariance matrix is only the
    # covariance of the residuals if we fit the intercept term
    what = np.linalg.lstsq(covariates, data)[0].T
    assert np.allclose(lr.weights, what)
    resid = data - covariates @ what.T
    assert np.allclose(lr.covariance_matrix, np.cov(resid, rowvar=False, bias=True), atol=1e-6)
