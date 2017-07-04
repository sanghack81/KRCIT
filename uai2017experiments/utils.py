import itertools

import numpy as np
import scipy.optimize
from numpy import diag
from numpy.core.umath import sqrt
from pygk.utils import repmat, as_column
from sdcit.synthetic_data import normalize
from sklearn.metrics import pairwise_distances
from uai2017experiments.cython_impl.cy_set_dist import *


def multiplys(*args):
    """Multiplying all matrices, None for empty"""
    temp = None
    for arg in args:
        if arg is not None:
            if temp is None:
                temp = arg.copy()
            else:
                temp *= arg
    return temp


def normalize_by_diag(k):
    if k is None:
        return None

    diag_array = diag(k).copy()
    diag_array[diag_array == 0.] = 1.  # will divide by 1
    x = 1. / sqrt(repmat(as_column(diag_array), 1, len(k)))
    k = (k * x) * x.transpose()
    return k


def __unroll(list_of_list):
    return list(itertools.chain(*list_of_list))


def c_kernel_matrix(data, index_of, VK, n_jobs, gamma, algorithm='r_convolution', equal_size_only=False,
                    ignore_values=False, ignore_structure=False):
    """Compute a kernel matrix with a given data and options.

    Parameters
    ----------
    data: array_like
        the data is an array of instances where an instance is a tuple of pairs of an item and its value.
    index_of: dict
        a mapping from items to indices
    VK: np.ndarray
        the pre-computed kernel matrix for items
    n_jobs: int
        the number of threads to use to compute the kernel matrix
    gamma: float
        gamma for a Gaussian RBF kernel
    algorithm: {'optimal_assignment','r_convolution'}
        algorithm of choice to deal with multiple values.
    equal_size_only: bool
        if True, kernel values between instances of different sizes will be 0.
    ignore_values: bool
        if True, values will be ignored (i.e., all values are treated as equal, distance between them is 0.)
    ignore_structure: bool
        if True, VK is removed (treated as a matrix of ones.

    """
    algorithm_codes = {alg: i for i, alg in enumerate(['r_convolution', 'optimal_assignment'])}

    if ignore_structure:
        item_data = np.array(__unroll([[0 for item, _ in d] for d in data]), dtype='int32', order='C')
        VK = np.ones((1, 1), dtype='float64')
    else:
        item_data = np.array(__unroll([[index_of[item] for item, _ in d] for d in data]), dtype='int32', order='C')

    if ignore_values:
        value_data = np.zeros(item_data.shape, dtype='float64', order='C')  # TODO can be faster
    else:
        value_data = np.array(__unroll([[value for _, value in d] for d in data]), dtype='float64', order='C')

    lengths = np.array([len(d) for d in data], dtype='int32', order='C')

    n = len(lengths)
    output = np.zeros((n, n), dtype='float64', order='C')

    equal_size_only = 1 if equal_size_only else 0

    cy_rpci(value_data, item_data, lengths, VK, output, n_jobs, gamma, algorithm_codes[algorithm], equal_size_only)

    return output


def auto_rbf_gamma(X, **unused):
    print('hope auto is not used ....')
    if X.ndim == 1:
        X = X[:, None]
    X = normalize(X)
    D = pairwise_distances(X)
    return kde_based_rbf_gamma(D)


def mle_score_cv_2(gamma, X, D2):
    LOWER_BOUND = 1e-10
    # adding
    train, test = X[:len(X) // 2], X[len(X) // 2:]
    K = np.exp(-gamma * D2[np.ix_(test, train)])  # kernel matrix
    return -np.sum(np.log(LOWER_BOUND + np.sum(K, axis=1) / np.sqrt(0.5 / gamma)))


def kde_based_rbf_gamma(D, nn=5, cutoff=None, init_x=1.0, repeat=5, seed=None, summarizer=np.median):
    """Compute the gamma parameter for a RBF kernel based on cross validation given an array of float numbers"""
    if seed is not None:
        np.random.seed(seed)
    if cutoff is None:  #
        cutoff = len(D) // 2
    idxs = np.arange(len(D))
    squared_D = D ** 2

    results = []
    for _ in range(repeat):
        Xs = [None] * nn
        for i in range(nn):
            np.random.shuffle(idxs)
            Xs[i] = idxs.copy() if len(idxs) <= cutoff else np.random.choice(idxs, cutoff, False)

        def func(ga):
            return sum(mle_score_cv_2(max(1e-10, ga[0]), X, squared_D) for X in Xs)

        res = scipy.optimize.minimize(func, np.array([init_x]), method='BFGS')
        results.append(res.x[0])
        init_x = res.x[0]
    # median is okay.
    # but min gives us an acceptable smooth(est) setting.
    if callable(summarizer):
        return summarizer(results)
    elif isinstance(summarizer, float):
        assert 0.0 <= float < 1.0
        return results[min(int(round(repeat * summarizer)), repeat - 1)]
    else:
        raise ValueError('unknown summarizer')


def AUPC(p_values) -> float:
    """Area Under Power Curve"""
    p_values = np.array(p_values)

    xys = [(uniq_v, np.mean(p_values <= uniq_v)) for uniq_v in np.unique(p_values)]

    area, prev_x, prev_y = 0, 0, 0
    for x, y in xys:
        area += (x - prev_x) * prev_y
        prev_x, prev_y = x, y

    area += (1 - prev_x) * prev_y
    return area
