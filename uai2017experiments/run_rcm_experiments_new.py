import itertools
import multiprocessing
import time
import warnings
from collections import defaultdict, Counter
from configparser import ConfigParser
from os.path import exists
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyrcds.domain import RelationalSchema, EntityClass, RelationshipClass, Cardinality, SkItem, remove_lone_entities
from pyrcds.domain import RelationalSkeleton
from pyrcds.model import enumerate_rpaths, RelationalDependency, RelationalVariable, RelationalPath, RCM, \
    enumerate_rvars, ParamRCM, generate_values_for_skeleton
from pyrcds.utils import linear_gaussian, normal_sampler, sum_agg
from sdcit.utils import random_seeds
from sklearn.metrics import euclidean_distances

from uai2017experiments.new_algos import ci_test_all


def generate_schema() -> RelationalSchema:
    """Fixed fully connected three entity classes."""
    # A with Z, B with Y, C with X
    A = EntityClass('A', 'Z')
    B = EntityClass('B', 'Y')
    C = EntityClass('C', 'X')

    es = [A, B, C]

    rs = [RelationshipClass('R_AB', [], {A: Cardinality.many, B: Cardinality.many}),
          RelationshipClass('R_AC', [], {A: Cardinality.many, C: Cardinality.many}),
          RelationshipClass('R_BC', [], {B: Cardinality.many, C: Cardinality.many})
          ]

    return RelationalSchema(es, rs)


def generate_skeleton(seed, schema: RelationalSchema, n_per_entity_class, avg_rel, max_rel) -> RelationalSkeleton:
    if seed is not None:
        np.random.seed(seed)

    # 1. initialize
    skeleton = RelationalSkeleton(schema, True)

    # 2. add entities
    entities = defaultdict(list)
    for E in schema.entities:
        for i in range(n_per_entity_class):
            item = SkItem(E.name + '_' + str(i).zfill(4), E)
            entities[E].append(item)
            skeleton.add_entity(item)

    assert len(schema.entities) == 3
    assert len(schema.relationships) == 3

    # 3. add relationships with 'correlation' using latent space.
    latent_pos = {E: np.random.rand(n_per_entity_class, 2) for E in schema.entities}

    # edge
    for R in schema.relationships:
        assert R.is_relationship_class
        E1, E2 = R.entities
        num_required_edges = int(n_per_entity_class * avg_rel)

        D = euclidean_distances(latent_pos[E1], latent_pos[E2], squared=True)
        K = np.exp(-50 * D)
        Pr = (K / np.sum(K)).flatten()
        selected12 = np.random.choice(len(Pr), size=3 * num_required_edges, replace=False, p=Pr)
        ent_pairs = np.array([[ij // n_per_entity_class, ij % n_per_entity_class] for ij in selected12], dtype=int)

        degree_counter1 = Counter()
        degree_counter2 = Counter()
        remain_edges = num_required_edges
        for i, (e1, e2) in enumerate(ent_pairs):
            if degree_counter1[e1] < max_rel and degree_counter2[e2] < max_rel:
                degree_counter1.update([e1])
                degree_counter2.update([e2])
                skeleton.add_relationship(SkItem(R.name + '_' + str(i).zfill(4), R),
                                          [entities[E1][e1], entities[E2][e2]])
                remain_edges -= 1
                if remain_edges <= 0:
                    break

        if remain_edges > 0:
            warnings.warn('times two is not enough?')

    return skeleton


def pick_one(arr):
    return arr[np.random.randint(len(arr))]


def generate_values(seed, schema, skeleton, null_hypothesis=True, mu=0.0, sd=0.1):
    if seed is not None:
        np.random.seed(seed)

    A, Z = schema['A'], schema['Z']
    B, Y = schema['B'], schema['Y']
    C, X = schema['C'], schema['X']
    R_AB = schema['R_AB']
    R_AC = schema['R_AC']
    R_BC = schema['R_BC']

    # 2-hop here = 1 hop in an entity-only graph
    rpaths_froms = {ent_class: list(enumerate_rpaths(schema, 2, ent_class)) for ent_class in [A, B, C]}

    # if X _||_Y | Z:
    #   (X --> Z --> Y) or (X <-- Z --> Y)
    # if not X _||_Y | Z
    #   (X --> Z <-- Y)
    independents = [
        [(X, Z), (Z, Y)],
        [(Z, X), (Z, Y)]
    ]
    dependents = [
        [(X, Z), (Y, Z)],
    ]

    if null_hypothesis:
        templates = independents
    else:
        templates = dependents

    # Model structure specification
    rdeps = []
    arrows = pick_one(templates)
    for from_attr, to_attr in arrows:
        # from_attr --> to_Attr
        from_item_class = schema.item_class_of(from_attr)
        base_item_class = schema.item_class_of(to_attr)
        rpath = pick_one(list(filter(lambda rp: rp.terminal == from_item_class, rpaths_froms[base_item_class])))

        cause_rvar = RelationalVariable(rpath, from_attr)
        effect_rvar = RelationalVariable(RelationalPath(base_item_class), to_attr)
        rdeps.append(RelationalDependency(cause_rvar, effect_rvar))

    rcm = RCM(schema, set(rdeps))

    # Model 'function' specification
    functions = dict()
    canonical_rvars = enumerate_rvars(schema, 0)
    for canonical_rvar in canonical_rvars:
        parents = list(rcm.pa(canonical_rvar))
        params = {pa_var: 1 for pa_var in parents}
        # noise = bias + noise (with 0 mean)
        functions[canonical_rvar] = linear_gaussian(params, sum_agg(), normal_sampler(mu, sd))

    param_rcm = ParamRCM(rcm.schema, rcm.directed_dependencies, functions)
    generate_values_for_skeleton(param_rcm, skeleton)

    if null_hypothesis:
        U = RelationalVariable(RelationalPath([B, R_BC, C]), X)
        V = RelationalVariable(RelationalPath(B), Y)  # canonical
        W = RelationalVariable(RelationalPath([B, R_AB, A]), Z)

    else:
        W = RelationalVariable(RelationalPath(A), Z)
        U = RelationalVariable(RelationalPath([A, R_AB, B]), Y)
        V = RelationalVariable(RelationalPath([A, R_AC, C]), X)

    return U, V, W, rcm, param_rcm


def krcit_data_gen(seed, n, avg_rel, max_rel, hypothesis):
    assert hypothesis in {'null', 'alternative'}
    np.random.seed(seed)

    schema = generate_schema()

    skeleton = generate_skeleton(random_seeds(), schema, n, avg_rel, max_rel)
    remove_lone_entities(skeleton)

    U, V, W, rcm, param_rcm = generate_values(random_seeds(), schema, skeleton, hypothesis == 'null', mu=0.3)

    # print("to normalize ? or not. normalization can be problematic if we want to use manually set gamma for RBF kernel based on blobs...")
    # normalize_skeleton(skeleton)

    return U, V, W, skeleton


def krcit_data_gen_twice(seed, n, avg_rel, max_rel, hypothesis):
    U, V, W, skeleton = krcit_data_gen(seed, n, avg_rel, max_rel, hypothesis)
    left = min(len(skeleton.items(skeleton.schema['A'])), len(skeleton.items(skeleton.schema['B'])),
               len(skeleton.items(skeleton.schema['C'])))

    U, V, W, skeleton = krcit_data_gen(seed, int(n * (n / left)), avg_rel, max_rel, hypothesis)
    left = min(len(skeleton.items(skeleton.schema['A'])), len(skeleton.items(skeleton.schema['B'])),
               len(skeleton.items(skeleton.schema['C'])))

    return U, V, W, skeleton, left


def dummy(seed, n, max_rel, hops, hypothesis):
    np.random.seed(seed)

    U, V, W, skeleton, actual_size = krcit_data_gen_twice(seed, n, max_rel / 2, max_rel, hypothesis)

    ps = ci_test_all(skeleton, U, V, W, vertex_kernel_hop=hops, gamma_x=50, gamma_y=50, gamma_z=50)

    return (seed, n, max_rel, hops, hypothesis, actual_size, *ps)


def unfinished_tasks(all_configurations, data_frame: pd.DataFrame, conf_columns: List[str]):
    done_configs = data_frame.as_matrix(conf_columns)
    done_configs = {tuple(config) for config in done_configs}

    all_as_set = {tuple(config) for config in all_configurations}
    assert done_configs <= all_as_set
    print('found {}/{} is done.'.format(len(set(done_configs) & all_as_set), len(all_as_set)))
    print('there are {} finished settings not in all_configurations.'.format(len(done_configs - all_as_set)))
    return list(all_as_set - done_configs)


def main():
    num_trials = 200
    expected_size_per_entity_class = [200, 400, 600, 800]  # data
    max_degree_per_relationship = [1, 2, 3]  # data
    graph_kernel_hops = [-1, 1, 2, 3, 4]  # test
    hypotheses = ['null', 'alternative']

    configs = set()
    configs |= set(itertools.product(expected_size_per_entity_class, [3], [1], hypotheses))
    configs |= set(itertools.product([800], max_degree_per_relationship, [1], hypotheses))
    configs |= set(itertools.product([800], [3], graph_kernel_hops, hypotheses))
    configs = sorted(configs)
    configs = [(i * num_trials + trial, *config) for trial in range(num_trials) for i, config in enumerate(configs)]

    filename = 'new_results/conditional.csv'
    if exists(filename):
        df = pd.read_csv(filename, names=['seed', 'n', 'max_rel', 'hops', 'hypothesis'], usecols=[0, 1, 2, 3, 4])
        configs = unfinished_tasks(configs, df, ['seed', 'n', 'max_rel', 'hops', 'hypothesis'])

    previous_batch_size = None
    previous_n_jobs = None
    start = time.time()
    total = len(configs)
    while configs:
        parser = ConfigParser()
        parser.read('run_rcm.ini')
        batch_size = int(parser['LINUX' if multiprocessing.cpu_count() == 32 else 'IMAC']['batch_size'])
        n_jobs = int(parser['LINUX' if multiprocessing.cpu_count() == 32 else 'IMAC']['n_jobs'])
        if previous_batch_size != batch_size:
            print('new batch size: {}'.format(batch_size))
        if previous_n_jobs != n_jobs:
            print('new num threads: {}'.format(n_jobs))
        previous_batch_size, previous_n_jobs = batch_size, n_jobs

        subconfigs = configs[:batch_size]
        configs = configs[batch_size:]
        outss = Parallel(n_jobs)(delayed(dummy)(*config) for config in subconfigs)
        with open(filename, 'a') as f:
            for outs in outss:
                print(*outs, sep=',', file=f)

        passed = time.time() - start
        remain = len(configs)
        avg_time_per_config = passed / (total - remain)
        remain_secs = avg_time_per_config * remain
        hours = remain_secs // 3600
        minutes = (remain_secs % 3600) // 60
        secs = remain_secs % 60
        print('{}h {}m {}s left'.format(int(hours), int(minutes), int(secs)))


if __name__ == '__main__':
    main()
