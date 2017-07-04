import numpy as np
from pyrcds.domain import RelationalSkeleton, AttributeClass
from pyrcds.model import RelationalVariable, SkeletonDataInterface
from sdcit.hsic import HSIC_boot
from sdcit.kcit import python_kcit_K
from sdcit.sdcit import SDCIT

from uai2017experiments.utils import normalize_by_diag, c_kernel_matrix
from uai2017experiments.algo import purge_data, infer_gammas
from uai2017experiments.kernel_utils import shortest_path_kernel_matrix
from uai2017experiments.residualize import SkResidualKernel, residualized_skeleton, residual_kernel_skeleton


def fetch_data(skeleton, U, V, W=None):
    """Fetch data represented by relational variables U, V, and W."""
    fetcher = SkeletonDataInterface(skeleton, to_shuffle=False)
    flatten_data = fetcher.flatten([U, V, W] if W is not None else [U, V], with_base_items=True)
    flatten_data = purge_data(flatten_data, True)

    base_items = flatten_data[:, 0]
    u_i_s = flatten_data[:, 1]
    v_i_s = flatten_data[:, 2]
    w_i_s = flatten_data[:, 3] if W is not None else None

    return base_items, u_i_s, v_i_s, w_i_s


def naive_kernel_ci_test(skeleton, U: RelationalVariable, V: RelationalVariable, W: RelationalVariable = None, *, gamma_x=None, gamma_y=None, gamma_z=None, use_median=False, X=None, Y=None, Z=None,
                         **unused):
    """Traditional kernel CI test based on R-convolution kernel."""
    if X is not None and Y is not None and (W is None or Z is not None):
        pass
    else:
        _, X, Y, Z = fetch_data(skeleton, U, V, W)
    gamma_x, gamma_y, gamma_z = infer_gammas(W, X, Y, Z, gamma_x, gamma_y, gamma_z, use_median)

    Kx = normalize_by_diag(c_kernel_matrix(X, None, None, 1, gamma_x, ignore_structure=True))
    Ky = normalize_by_diag(c_kernel_matrix(Y, None, None, 1, gamma_y, ignore_structure=True))
    if W is not None:
        Kz = normalize_by_diag(c_kernel_matrix(Z, None, None, 1, gamma_z, ignore_structure=True))

    if W is None:
        temp = HSIC_boot(Kx, Ky, num_boot=400)
        return temp, temp
    else:
        if len(Kx) % 4:
            n = len(Kx)
            new_n = n - (n % 4)
            Kx = Kx[:new_n, :new_n]
            Ky = Ky[:new_n, :new_n]
            Kz = Kz[:new_n, :new_n]

        return python_kcit_K(Kx, Ky, Kz)[2], SDCIT(Kx, Ky, Kz, size_of_null_sample=400)[1]


def r_conv_with_precomputed_ks(X, info: SkResidualKernel, x_attr: AttributeClass):
    assert x_attr is not None and isinstance(x_attr, AttributeClass)
    tuples_of_pairs = X
    n = len(tuples_of_pairs)

    Kx = np.zeros((n, n))

    for i, tuple_i in enumerate(tuples_of_pairs):
        for j, tuple_j in enumerate(tuples_of_pairs[i:], start=i):
            temp = 0
            for item_ia, _ in tuple_i:
                for item_jb, _ in tuple_j:
                    temp += info.k(item_ia, item_jb, x_attr)
            Kx[i, j] = Kx[j, i] = temp
    return Kx


def with_graph_kernel_ci_test(skeleton: RelationalSkeleton, U: RelationalVariable, V: RelationalVariable, W: RelationalVariable = None, *, VK, index_of, gamma_x=None, gamma_y=None, gamma_z=None,
                              use_median=False, X=None, Y=None, Z=None, **unused):
    """Residual-kernel-based CI test based on R-convolution kernel."""
    if X is not None and Y is not None and (W is None or Z is not None):
        pass
    else:
        _, X, Y, Z = fetch_data(skeleton, U, V, W)

    gamma_x, gamma_y, gamma_z = infer_gammas(W, X, Y, Z, gamma_x, gamma_y, gamma_z, use_median)  # # skeleton to entities-only unidrected graph

    # -- compute kernel matrices
    K_X = normalize_by_diag(c_kernel_matrix(X, index_of, VK, 1, gamma_x))
    K_Y = normalize_by_diag(c_kernel_matrix(Y, index_of, VK, 1, gamma_y))
    K_Z = normalize_by_diag(c_kernel_matrix(Z, index_of, VK, 1, gamma_z)) if W is not None else np.ones(K_Y.shape)

    K_X2 = normalize_by_diag(c_kernel_matrix(X, index_of, VK, 1, 1.0, ignore_values=True))
    K_Y2 = normalize_by_diag(c_kernel_matrix(Y, index_of, VK, 1, 1.0, ignore_values=True))
    K_Z2 = normalize_by_diag(c_kernel_matrix(Z, index_of, VK, 1, 1.0, ignore_values=True)) if W is not None else 1
    K_ZG = K_Z * K_X2 * K_Y2 * K_Z2

    if len(K_X) % 4:
        n = len(K_X)
        new_n = n - (n % 4)
        K_X = K_X[:new_n, :new_n]
        K_Y = K_Y[:new_n, :new_n]
        K_ZG = K_ZG[:new_n, :new_n]

    return python_kcit_K(K_X, K_Y, K_ZG)[2], SDCIT(K_X, K_Y, K_ZG, size_of_null_sample=400)[1]


def residual_kernel_kernel_ci_test(skeleton: RelationalSkeleton, residual_skeleton_kernel_info: SkResidualKernel, U: RelationalVariable, V: RelationalVariable, W: RelationalVariable = None, X=None,
                                   Y=None, Z=None, **unused):
    """Residual-kernel-based CI test based on R-convolution kernel."""
    if X is not None and Y is not None and (W is None or Z is not None):
        pass
    else:
        _, X, Y, Z = fetch_data(skeleton, U, V, W)
    Kx = normalize_by_diag(r_conv_with_precomputed_ks(X, residual_skeleton_kernel_info, U.attr))
    Ky = normalize_by_diag(r_conv_with_precomputed_ks(Y, residual_skeleton_kernel_info, V.attr))
    if W is not None:
        Kz = normalize_by_diag(r_conv_with_precomputed_ks(Z, residual_skeleton_kernel_info, W.attr))

    if W is None:
        temp = HSIC_boot(Kx, Ky, num_boot=400)
        return temp, temp
    else:
        if len(Kx) % 4:
            n = len(Kx)
            new_n = n - (n % 4)
            Kx = Kx[:new_n, :new_n]
            Ky = Ky[:new_n, :new_n]
            Kz = Kz[:new_n, :new_n]

        return python_kcit_K(Kx, Ky, Kz)[2], SDCIT(Kx, Ky, Kz, size_of_null_sample=400)[1]


def ci_test_all(skeleton: RelationalSkeleton, U, V, W=None, *, vertex_kernel_hop=2, gamma_x=None, gamma_y=None, gamma_z=None, use_median=False):
    VK, index_of = shortest_path_kernel_matrix(skeleton, vertex_kernel_hop)
    VK = normalize_by_diag(VK)

    _, X, Y, Z = fetch_data(skeleton, U, V, W)
    kwargs = {'gamma_x': gamma_x, 'gamma_y': gamma_y, 'gamma_z': gamma_z, 'use_median': use_median}

    # -- Naive
    naive_p_kcit, naive_p_sdcit = naive_kernel_ci_test(skeleton, U, V, W, X=X, Y=Y, Z=Z, **kwargs)

    # -- residualized (always use median!)
    res_skeleton = residualized_skeleton(skeleton, VK, index_of)
    res_p_kcit, res_p_sdcit = naive_kernel_ci_test(res_skeleton, U, V, W, use_median=True)

    # -- residual kernel
    info = residual_kernel_skeleton(skeleton, VK, index_of)
    reskern_p_kcit, reskern_p_sdcit = residual_kernel_kernel_ci_test(skeleton, info, U, V, W, X=X, Y=Y, Z=Z)

    # -- graph-explicit
    graph_p_kcit, graph_p_sdcit = with_graph_kernel_ci_test(skeleton, U, V, W, VK=VK, index_of=index_of, X=X, Y=Y, Z=Z, **kwargs)

    return naive_p_kcit, naive_p_sdcit, res_p_kcit, res_p_sdcit, reskern_p_kcit, reskern_p_sdcit, graph_p_kcit, graph_p_sdcit
