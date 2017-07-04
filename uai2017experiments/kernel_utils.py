import networkx as nx
from pygk.labeled import labeled_shortest_path_kernel
from pygk.utils import KGraph
from pyrcds.domain import RelationalSkeleton, skeleton_to_entities_only_nx_graph


# TODO VK only for the same item class!
def shortest_path_kernel_matrix(skeleton: RelationalSkeleton, vertex_kernel_hop: int):
    """a precomputed kernel matrix-based shortest path kernel function given a k-hop neighborhood subgraph"""
    entities_ug = skeleton_to_entities_only_nx_graph(skeleton)
    entities_ordered = list(entities_ug.nodes())
    kgraphs = [None] * len(entities_ordered)
    for i, entity in enumerate(entities_ordered):
        reachable = list(nx.single_source_shortest_path_length(entities_ug, entity, vertex_kernel_hop).keys())
        kgraphs[i] = KGraph(nx.subgraph(entities_ug, reachable), 'item_class')

    index_of = {item: index for index, item in enumerate(entities_ordered)}
    VK, _ = labeled_shortest_path_kernel(kgraphs)

    return VK, index_of
