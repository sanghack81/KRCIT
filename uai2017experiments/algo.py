from typing import List

import numpy as np
import numpy.ma as ma
from pyrcds.domain import RelationalSkeleton
from pyrcds.model import RelationalVariable, SkeletonDataInterface
from sklearn.metrics import euclidean_distances

from uai2017experiments.kernel_utils import shortest_path_kernel_matrix
from uai2017experiments.utils import auto_rbf_gamma, normalize_by_diag, c_kernel_matrix

RVARS = List[RelationalVariable]


def purge_data(flatten_data, with_base_items):
    """Remove rows with an empty cell (terminal set is empty)"""
    to_remove = set()
    for i in range(1 if with_base_items else 0, flatten_data.shape[1]):
        vals = flatten_data[:, i]
        for row, val in enumerate(vals):
            if not val:
                to_remove.add(row)
    remained = list(set(range(len(flatten_data))) - to_remove)
    remained = sorted(remained)
    return flatten_data[remained, :]


def get_KZG(skeleton: RelationalSkeleton, vertex_kernel_hop, U, V, W=None, ignore_gk=False):
    # Compute kernel matrices
    # -- fetch
    fetcher = SkeletonDataInterface(skeleton, to_shuffle=False)
    flatten_data = fetcher.flatten([U, V, W] if W is not None else [U, V], with_base_items=True)
    flatten_data = purge_data(flatten_data, True)
    u_i_s = flatten_data[:, 1]
    v_i_s = flatten_data[:, 2]
    w_i_s = flatten_data[:, 3] if W is not None else None

    # -- vertex kernel (through graph kernel)
    VK, index_of = prep_VK(ignore_gk, skeleton, vertex_kernel_hop)

    # -- RBF kernel parameter
    gamma_x, gamma_y, gamma_z = infer_gammas(W, u_i_s, v_i_s, w_i_s)

    # -- compute kernel matrices
    K_Z = normalize_by_diag(c_kernel_matrix(w_i_s, index_of, VK, 1, gamma_z, ignore_structure=ignore_gk)) if W is not None else np.ones((len(u_i_s), len(u_i_s)))

    if ignore_gk:
        K_ZG = K_Z
    else:
        K_X2 = normalize_by_diag(c_kernel_matrix(u_i_s, index_of, VK, 1, gamma_x, ignore_values=True, ignore_structure=ignore_gk))
        K_Y2 = normalize_by_diag(c_kernel_matrix(v_i_s, index_of, VK, 1, gamma_y, ignore_values=True, ignore_structure=ignore_gk))
        K_Z2 = normalize_by_diag(c_kernel_matrix(w_i_s, index_of, VK, 1, gamma_z, ignore_values=True, ignore_structure=ignore_gk)) if W is not None else 1
        K_ZG = K_Z * K_X2 * K_Y2 * K_Z2

    return K_ZG


def prep_VK(ignore_gk, skeleton, vertex_kernel_hop, VK=None, index_of=None):
    if ignore_gk:
        index_of = None
        VK = None
    else:
        if VK is None:
            VK, index_of = shortest_path_kernel_matrix(skeleton, vertex_kernel_hop)
            VK = normalize_by_diag(VK)
    return VK, index_of


def med_except_diag(x):
    D_squared = euclidean_distances(x, squared=True)
    # masking upper triangle and the diagonal.
    mask = np.triu(np.ones(D_squared.shape), 0)
    median_squared_distance = ma.median(ma.array(D_squared, mask=mask))
    return median_squared_distance


def infer_gammas(W, u_i_s, v_i_s, w_i_s, gamma_x=None, gamma_y=None, gamma_z=None, use_median=False):
    if gamma_x is None:
        xx = np.array(list({pair[1] for tuple_of_pairs in u_i_s for pair in tuple_of_pairs}))
        if use_median:
            gamma_x = 0.5 / med_except_diag(xx[:, None])
        else:
            gamma_x = auto_rbf_gamma(xx)

    if gamma_y is None:
        yy = np.array(list({pair[1] for tuple_of_pairs in v_i_s for pair in tuple_of_pairs}))
        if use_median:
            gamma_y = 0.5 / med_except_diag(yy[:, None])
        else:
            gamma_y = auto_rbf_gamma(yy)

    if gamma_z is None and W is not None:
        zz = np.array(list({pair[1] for tuple_of_pairs in w_i_s for pair in tuple_of_pairs}))
        if use_median:
            gamma_z = 0.5 / med_except_diag(zz[:, None])
        else:
            gamma_z = auto_rbf_gamma(zz)

    return gamma_x, gamma_y, gamma_z
