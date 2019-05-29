import numpy as np
from gpflow.models import GPR
from gpflow.kernels import White, Linear
from pyrcds.domain import RelationalSkeleton, SkItem
from sdcit.utils import pdinv, rbf_kernel_median, residual_kernel, truncated_eigen, eigdec

from uai2017experiments.utils import normalize_by_diag
from uai2017experiments.kernel_utils import shortest_path_kernel_matrix


def compute_residual_eig(Y: np.ndarray, Kx: np.ndarray) -> np.ndarray:
    """Residual of Y based on Kx, a kernel matrix of X"""
    assert len(Y) == len(Kx)

    eig_Kx, eix = truncated_eigen(*eigdec(Kx, min(100, len(Kx) // 4)))
    phi_X = eix @ np.diag(np.sqrt(eig_Kx))  # X @ X.T is close to K_X
    n_feats = phi_X.shape[1]

    linear_kernel = Linear(n_feats, ARD=True)
    gp_model = GPR(phi_X, Y, linear_kernel + White(n_feats))
    gp_model.optimize()

    new_Kx = linear_kernel.compute_K_symm(phi_X)
    sigma_squared = gp_model.kern.white.variance.value[0]

    return (pdinv(np.eye(len(Kx)) + new_Kx / sigma_squared) @ Y).squeeze()


def residualize_skeleton(skeleton: RelationalSkeleton, vertex_kernel_hop=2, normalize_vk=True):
    """Replace values with residuals based on graph kernel"""
    print('this changes values!!!! use copying one...')

    # Prepare vertex kernel matrix
    VK, index_of = shortest_path_kernel_matrix(skeleton, vertex_kernel_hop)
    if normalize_vk:
        VK = normalize_by_diag(VK)

    #
    schema = skeleton.schema
    for item_class in schema.item_classes:
        items = list(skeleton.items(item_class))
        if not item_class.attrs:
            continue

        item_indices_on_VK = [index_of[item] for item in items]
        for attr_class in item_class.attrs:
            values = np.array([skeleton[(item, attr_class)] for item in items])
            subvk = VK[np.ix_(item_indices_on_VK, item_indices_on_VK)]
            to_replace = compute_residual_eig(values[:, None], subvk)
            for item, rep_val in zip(items, to_replace):
                skeleton[(item, attr_class)] = rep_val


def copy_skeleton_structure(skeleton: RelationalSkeleton) -> RelationalSkeleton:
    new_skeleton = RelationalSkeleton(skeleton.schema)
    mapper = dict()
    for ent_class in skeleton.schema.entities:
        for item in skeleton.items(ent_class):
            assert isinstance(item, SkItem)
            new_item = SkItem(item.name, item.item_class)
            mapper[item] = new_item
            new_skeleton.add_entity(new_item)
    for rel_class in skeleton.schema.relationships:
        for item in skeleton.items(rel_class):
            assert isinstance(item, SkItem)
            new_skeleton.add_relationship(SkItem(item.name, item.item_class),
                                          [mapper[ne] for ne in skeleton.neighbors(item)])
    return new_skeleton, mapper


def residualized_skeleton(skeleton: RelationalSkeleton, VK, index_of):
    """Replace values with residuals based on graph kernel"""
    new_skeleton, mapper = copy_skeleton_structure(skeleton)

    #
    schema = skeleton.schema
    for item_class in schema.item_classes:
        items = list(skeleton.items(item_class))
        if not item_class.attrs:
            continue

        item_indices_on_VK = [index_of[item] for item in items]
        for attr_class in item_class.attrs:
            values = np.array([skeleton[(item, attr_class)] for item in items])
            subvk = VK[np.ix_(item_indices_on_VK, item_indices_on_VK)]
            to_replace = compute_residual_eig(values[:, None], subvk)
            for item, rep_val in zip(items, to_replace):
                new_skeleton[(mapper[item], attr_class)] = rep_val
    return new_skeleton


class SkResidualKernel:
    def __init__(self, out):
        self.out = out

    def k(self, item_attr1, item_attr2, attr=None):
        if attr is not None:
            item1 = item_attr1
            item2 = item_attr2
            lookup, K = self.out[attr]
            return K[lookup[item1], lookup[item2]]

        item1 = item_attr1[0]
        item2 = item_attr2[0]
        attr = item_attr1[1]
        assert attr == item_attr2[1]
        lookup, K = self.out[attr]
        return K[lookup[item1], lookup[item2]]


def residual_kernel_skeleton(skeleton: RelationalSkeleton, VK, index_of):
    """Residual kernel and relevant information for every attribute class"""
    out = dict()

    schema = skeleton.schema
    for item_class in schema.item_classes:
        items = list(skeleton.items(item_class))
        if not item_class.attrs:
            continue

        item_indices_on_VK = [index_of[item] for item in items]
        lookup = {item: i for i, item in enumerate(items)}
        for attr_class in item_class.attrs:
            values = np.array([skeleton[(item, attr_class)] for item in items])
            subvk = VK[np.ix_(item_indices_on_VK, item_indices_on_VK)]

            K = rbf_kernel_median(values[:, None])
            K2 = residual_kernel(K, subvk)
            out[attr_class] = (lookup, K2)

    return SkResidualKernel(out)
