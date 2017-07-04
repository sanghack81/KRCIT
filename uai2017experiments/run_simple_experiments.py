import itertools
from itertools import chain

import numpy as np
from numpy.random import permutation
from pyrcds.domain import RelationalSchema, EntityClass, RelationshipClass, Cardinality, RelationalSkeleton, SkItem
from pyrcds.model import RelationalVariable, RelationalPath, RCM, RelationalDependency, ParamRCM, \
    generate_values_for_skeleton, terminal_set, SkeletonDataInterface
from pyrcds.utils import average_agg, sum_agg
from pyrcds.utils import normal_sampler, linear_gaussian
from sdcit.sdcit import permuted
from sdcit.utils import K2D

from uai2017experiments.algo import get_KZG

A, B, C, D = es = [EntityClass('A', 'X'), EntityClass('B', 'Y'), EntityClass('C', 'U'), EntityClass('D', 'V')]
X, Y, U, V = next(iter(A.attrs)), next(iter(B.attrs)), next(iter(C.attrs)), next(iter(D.attrs))

AB, AC, BD = rs = [RelationshipClass('R_AB', [], {A: Cardinality.one, B: Cardinality.one}),
                   RelationshipClass('R_AC', [], {A: Cardinality.many, C: Cardinality.many}),
                   RelationshipClass('R_BD', [], {B: Cardinality.many, D: Cardinality.many})]

schema = RelationalSchema(es, rs)


def simple_random_test_inspect(seed, structure_random_p, n, mu, sd, independent, vertex_kernel_hop, slope, fname,
                               titlestr):
    np.random.seed(seed)
    # np random seed?
    U = RelationalVariable(RelationalPath([A]), X)
    V = RelationalVariable(RelationalPath([A, AB, B]), Y)
    W = None

    # 1. Structuring
    skeleton = generate_structure(n, structure_random_p)

    # 2. Values
    if independent:
        slope = 0

    generate_values(independent, mu, sd, skeleton, slope)

    K_ZG0 = get_KZG(skeleton, vertex_kernel_hop, U, V, W, ignore_gk=False)
    K_ZG1 = get_KZG(skeleton, vertex_kernel_hop, U, V, W, ignore_gk=True)

    fetcher = SkeletonDataInterface(skeleton, to_shuffle=False)
    flatten_data = fetcher.flatten([U, V], with_base_items=True)
    base_items = flatten_data[:, 0]
    u_i_s = flatten_data[:, 1]
    v_i_s = flatten_data[:, 2]

    x_values = np.array([pair[1] for tuple_of_pairs in u_i_s for pair in tuple_of_pairs])
    y_values = np.array([pair[1] for tuple_of_pairs in v_i_s for pair in tuple_of_pairs])

    PP0 = permuted(K2D(K_ZG0))
    PP1 = permuted(K2D(K_ZG1))
    yy0 = y_values[PP0]
    yy1 = y_values[PP1]

    import pandas as pd
    import seaborn
    import matplotlib.pyplot as plt
    df0 = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    # don't apply to permutation on 'cc'. This is to preserve colors!
    df0['x'] = x_values
    df0['y'] = y_values
    df0['cc'] = 2 * np.array([len(skeleton.neighbors(item, AC)) for item in base_items]) + np.array(
        [len(skeleton.neighbors(list(terminal_set(skeleton, RelationalPath([A, AB, B]), item))[0], BD)) for item in
         base_items])
    df0['type'] = 'given'

    df1['x'] = x_values
    df1['y'] = yy0
    df1['cc'] = df0['cc']
    df1['type'] = 'with GK'  # graph-kernel

    df2['x'] = x_values
    df2['y'] = yy1
    df2['cc'] = df0['cc']
    df2['type'] = 'without GK'  # graph-kernel

    df = pd.concat([df0, df1, df2], ignore_index=True)

    seaborn.set(style='white', font_scale=1.3, palette=seaborn.color_palette('Set1', 4))
    plt.figure(figsize=[4, 4])
    g = seaborn.FacetGrid(df, col="type", hue='cc', hue_order=[0, 3, 2, 1], size=3, aspect=1)
    g.map(plt.scatter, "x", "y", alpha=0.5, linewidth='0')

    titles = [' ', ' ', ' ']
    for ax, title in zip(g.axes.flat, titles):
        ax.set_title(title)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.suptitle(titlestr)
    plt.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def generate_values(independent, mu, sd, skeleton, slope, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if independent:
        slope = 0.0

    var_X = RelationalVariable(RelationalPath(A), X)
    var_Y = RelationalVariable(RelationalPath(B), Y)
    var_U = RelationalVariable(RelationalPath(C), U)
    var_V = RelationalVariable(RelationalPath(D), V)
    rcm = RCM(schema,
              {
                  RelationalDependency(RelationalVariable(RelationalPath([A, AC, C]), U), var_X),
                  RelationalDependency(RelationalVariable(RelationalPath([B, BD, D]), V), var_Y),
                  RelationalDependency(RelationalVariable(RelationalPath([B, AB, A]), X), var_Y)  # X-->Y
              }
              )

    functions = {
        var_U: linear_gaussian(dict(), average_agg(), normal_sampler(mu, sd)),
        var_V: linear_gaussian(dict(), average_agg(), normal_sampler(mu, sd)),
        var_X: linear_gaussian({RelationalVariable(RelationalPath([A, AC, C]), U): 1.0}, sum_agg(),
                               normal_sampler(0, sd)),
        var_Y: linear_gaussian({RelationalVariable(RelationalPath([B, BD, D]), V): 1.0,
                                RelationalVariable(RelationalPath([B, AB, A]), X): slope}, sum_agg(),
                               normal_sampler(0, sd))
    }

    # Parametrize RCM and generate values
    param_rcm = ParamRCM(rcm.schema, rcm.directed_dependencies, functions)
    generate_values_for_skeleton(param_rcm, skeleton)


# This ensures fully separated connected components
def generate_structure(num_entities_per_class, randomness=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    skeleton = RelationalSkeleton(schema, True)
    A_ents = [SkItem(A.name + '_' + str(i).zfill(4), A) for i in range(num_entities_per_class)]
    B_ents = [SkItem(B.name + '_' + str(i).zfill(4), B) for i in range(num_entities_per_class)]
    C_ents = [SkItem(C.name + '_' + str(i).zfill(4), C) for i in range(num_entities_per_class // 2)]
    D_ents = [SkItem(D.name + '_' + str(i).zfill(4), D) for i in range(num_entities_per_class // 2)]

    for e in chain(A_ents, B_ents, C_ents, D_ents):
        skeleton.add_entity(e)

    ACs = [SkItem(AC.name + '_' + str(i).zfill(4), AC) for i in range(num_entities_per_class // 2)]
    BDs = [SkItem(BD.name + '_' + str(i).zfill(4), BD) for i in range(num_entities_per_class // 2)]

    # [0, n//2) for no parents
    # [n//2, n) for 1 parent
    for i in range(num_entities_per_class // 2):
        skeleton.add_relationship(ACs[i], [A_ents[i], C_ents[i]])
        skeleton.add_relationship(BDs[i], [B_ents[i], D_ents[i]])

    # n matches??
    ABs = [SkItem(AB.name + '_' + str(i).zfill(4), AB) for i in range(num_entities_per_class)]

    # Start with biased relationship
    # biased
    mapping = list(range(num_entities_per_class))
    to_move_idx = np.random.choice(num_entities_per_class, int(num_entities_per_class * randomness), replace=False)
    new_target = permutation(to_move_idx)
    for i, j in zip(to_move_idx, new_target):
        mapping[i] = j

    for i in range(num_entities_per_class):
        skeleton.add_relationship(ABs[i], [A_ents[i], B_ents[mapping[i]]])

    count = np.zeros([2, 2])
    for ab in skeleton.items(AB):
        i_item = next(iter(skeleton.neighbors(ab, A)))
        j_item = next(iter(skeleton.neighbors(ab, B)))
        x = len(skeleton.neighbors(i_item, AC))
        y = len(skeleton.neighbors(j_item, BD))
        count[x, y] += 1

    return skeleton


def null_configurations2():
    configurations = []
    for mixup in np.linspace(0, 1, 10 + 1):
        for mu in np.linspace(0, 1, 10 + 1):
            configurations.append((mixup, mu))
    return configurations


def alternative_configurations2(random_structure, mu, vertex_kernel_hop=2):
    configurations = []
    n = 200
    for slope in np.linspace(0, 0.7, 14 + 1):
        configurations.append((random_structure, n, mu, 0.1, False, vertex_kernel_hop, slope))
    return configurations


def main():
    # draw how null hypothesis is perceived.
    for dependent, homo, biased in itertools.product([True, False], repeat=3):
        simple_random_test_inspect(seed=0,  # seed
                                   structure_random_p=0 if biased else 1,  # randomized == 1
                                   n=200,  # n
                                   mu=0.0 if homo else 0.7,  # mu
                                   sd=0.1,  # sd
                                   independent=not dependent,  # independent
                                   vertex_kernel_hop=2,  # vertex kernel hop
                                   slope=1.0 if dependent else 0.0,  # slope
                                   fname='new_figures/vis_{}_{}{}.pdf'.format(
                                       'dependent' if dependent else 'independent',
                                       'homo' if homo else 'hetero',
                                       '_bias' if biased else ''),
                                   titlestr='{} & {}{}'.format('Alternative' if dependent else 'Null',
                                                               'Homogeneous' if homo else 'Heterogeneous',
                                                               ' (biased)' if biased else ' (randomized)'))


if __name__ == '__main__':
    main()
