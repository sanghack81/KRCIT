from itertools import chain

import numpy as np
from numpy.random import permutation
from pyrcds.domain import RelationalSchema, EntityClass, RelationshipClass, Cardinality, RelationalSkeleton, SkItem, \
    remove_lone_entities

A, B, C, D = es = [EntityClass('A', 'X'), EntityClass('B', 'Y'), EntityClass('C', 'U'), EntityClass('D', 'V')]
X, Y, _, _ = next(iter(A.attrs)), next(iter(B.attrs)), next(iter(C.attrs)), next(iter(D.attrs))

AB, AC, BD = rs = [RelationshipClass('R_AB', [], {A: Cardinality.many, B: Cardinality.many}),
                   RelationshipClass('R_AC', [], {A: Cardinality.many, C: Cardinality.many}),
                   RelationshipClass('R_BD', [], {B: Cardinality.many, D: Cardinality.many})]

schema = RelationalSchema(es, rs)


# This ensures fully separated connected components
def generate_structure(n, random_rewiring=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    skeleton = RelationalSkeleton(schema, True)
    A_ents = [SkItem(A.name + '_' + str(i).zfill(4), A) for i in range(n)]
    B_ents = [SkItem(B.name + '_' + str(i).zfill(4), B) for i in range(len(A_ents))]

    C_ents = [SkItem(C.name + '_' + str(i).zfill(4), C) for i in range(int(0.5 * len(A_ents)))]  # 0, 1, 2, 3 ...
    D_ents = [SkItem(D.name + '_' + str(i).zfill(4), D) for i in range(len(C_ents))]
    for e in chain(A_ents, B_ents, C_ents, D_ents):
        skeleton.add_entity(e)

    count = 0
    # average connection is 2.
    # biased C & D correlate!
    # choose A for each C, twice!
    for i in range(len(A_ents)):
        c1, c2, c3 = np.random.choice(len(C_ents), 3, replace=False)
        rel = SkItem(AC.name + '_' + str(count).zfill(4), AC)
        count += 1
        skeleton.add_relationship(rel, [A_ents[i], C_ents[c1]])

        if i > len(A_ents) / 3:
            rel = SkItem(AC.name + '_' + str(count).zfill(4), AC)
            count += 1
            skeleton.add_relationship(rel, [A_ents[i], C_ents[c2]])

        if i > 2 * len(A_ents) / 3:
            rel = SkItem(AC.name + '_' + str(count).zfill(4), AC)
            count += 1
            skeleton.add_relationship(rel, [A_ents[i], C_ents[c3]])

    count = 0
    for i in range(len(B_ents)):
        d1, d2, d3 = np.random.choice(len(D_ents), 3, replace=False)
        rel = SkItem(BD.name + '_' + str(count).zfill(4), BD)
        count += 1
        skeleton.add_relationship(rel, [B_ents[i], D_ents[d1]])

        if i > len(B_ents) / 3:
            rel = SkItem(BD.name + '_' + str(count).zfill(4), BD)
            count += 1
            skeleton.add_relationship(rel, [B_ents[i], D_ents[d2]])

        if i > 2 * len(B_ents) / 3:
            rel = SkItem(BD.name + '_' + str(count).zfill(4), BD)
            count += 1
            skeleton.add_relationship(rel, [B_ents[i], D_ents[d3]])

    # n matches??
    ABs = [SkItem(AB.name + '_' + str(i).zfill(4), AB) for i in range(2 * len(A_ents))]
    already = set()
    # biased
    mapping = list(range(n))
    to_move_idx = np.random.choice(n, int(n * random_rewiring), replace=False)
    new_target = permutation(to_move_idx)
    for i, j in zip(to_move_idx, new_target):
        mapping[i] = j

    for i in range(n):
        already.add((i, mapping[i]))
        skeleton.add_relationship(ABs[i], [A_ents[i], B_ents[mapping[i]]])

    for i in range(n):
        aindex = np.random.randint(len(A_ents))
        bindex = np.random.randint(len(B_ents))
        if (aindex, bindex) in already:
            continue
        else:
            already.add((aindex, bindex))
        skeleton.add_relationship(ABs[i + n], [A_ents[aindex], B_ents[bindex]])

    remove_lone_entities(skeleton)

    return skeleton
