import inspect
from functools import wraps, partial
from enum import IntEnum

import jax.numpy as np
from jax.flatten_util import ravel_pytree

class Verbosity(IntEnum):
    OFF = 0
    QUIET = 1
    LOUD = 2
    DEBUG = 3


def format_dataset(f):
    sig = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        # Get the `dataset` argument
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        dataset = bound_args.arguments['dataset']

        # Make sure dataset is a list of dictionaries
        if isinstance(dataset, (list, tuple)):
            assert all([isinstance(d, dict) and "data" in d for d in dataset])
        elif isinstance(dataset, dict):
            assert "data" in dataset
            dataset = [dataset]
        elif isinstance(dataset, np.ndarray):
            dataset = [dict(data=dataset)]
        else:
            raise Exception("Expected `dataset` to be a numpy array, a dictionary, or a "
                            "list of dictionaries.  See help(ssm.HMM) for more details.")

        # Update the bound arguments
        bound_args.arguments['dataset'] = dataset

        # Make sure `weights` is a list of ones like the dataset, if present
        if 'weights' in bound_args.arguments:
            weights = bound_args.arguments['weights']

            if isinstance(weights, (list, tuple)):
                assert all([len(w) == len(data_dict["data"])
                            for w, data_dict in zip(weights, dataset)])
            elif weights is None:
                weights = [np.ones(len(data_dict['data']))
                           for data_dict in dataset]
            else:
                # Assume weights is 'array like'
                assert len(dataset) == 1
                assert len(weights) == len(dataset[0]['data'])
                weights = [weights]

            # Update the bound args
            bound_args.arguments['weights'] = weights

        # Call the function
        return f(*bound_args.args, **bound_args.kwargs)

    return wrapper


def sum_tuples(a, b):
    assert a or b
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return tuple(ai + bi for ai, bi in zip(a, b))


def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K))
    zoh[np.arange(N), np.arange(K)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K,))
    return zoh


def convex_combination(curr_params, new_params, step_size):
    """
    Output next = step_size * target + (1-step_size) * curr
    where target, curr, and next can be PyTree's of nested
    containers with arrays/scalars at the leaves.
    Assume curr and target have the same structure.
    """
    # assert step_size >= 0 and step_size <= 1
    _curr_params, unravel = ravel_pytree(curr_params)
    _new_params, _ = ravel_pytree(new_params)
    return unravel((1 - step_size) * _curr_params + step_size * _new_params)
