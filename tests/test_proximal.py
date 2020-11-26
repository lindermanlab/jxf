import time
import jax.numpy as np
import jax.random as jr
import jxf.distributions as dists

def test_normal(shape=(10000,), num_epochs=25, batch_size=100):
    assert shape[0] > batch_size
    num_batches = shape[0] // batch_size
    key = jr.PRNGKey(time.time_ns())
    data = 5 * jr.normal(key, shape=shape)

    schedule = lambda itr: (1 + itr)**(-0.75)

    state, update, get_dist = dists.Normal.proximal_optimizer(step_size=schedule)

    for itr in range(num_epochs):
        for batch_itr in range(num_batches):
            offset = batch_itr * batch_size
            state = update(itr * num_batches + batch_itr,
                           data[offset:offset+batch_size],
                           state,
                           scale_factor=shape[0] / batch_size)
            norm = get_dist(state)
        print("epoch:", itr, "loc:", norm.loc, "scale:", norm.scale)

    assert np.allclose(data.mean(), norm.loc, atol=1e-2)
    assert np.allclose(data.std(), norm.scale, atol=1e-2)

if __name__ == "__main__":
    test_normal()
