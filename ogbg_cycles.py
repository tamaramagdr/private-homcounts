import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
import networkx as nx
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from sbm import cycle_hom_densities_and_counts


def get_ogbg_cycle_hom_features(k_max=10, dataset='molhiv'):
    dataset = PygGraphPropPredDataset(name=f'ogbg-{dataset}')

    # Compute cycle hom counts and densities for all graphs in the dataset.
    ks = list(range(3, k_max + 1))
    pattern_sizes = np.array(ks, dtype=int)
    pattern_edges = np.array(ks, dtype=int)

    # Get adjacency matrices from dataset.
    adj_matrices = []
    all_homdensities = []
    all_homcounts = []
    graph_sizes = []
    for i in tqdm(range(len(dataset)), desc="Processing graphs"):
        data = dataset[i]
        nx_graph = to_networkx(data).to_undirected()
        adj_matrix = nx.to_numpy_array(nx_graph)
        adj_matrices.append(adj_matrix)
        hom_densities, hom_counts = cycle_hom_densities_and_counts(adj_matrix, ks=ks)
        all_homdensities.append(hom_densities)
        all_homcounts.append(hom_counts)
        graph_sizes.append(adj_matrix.shape[0])

    return dict(
        hom_densities=np.vstack(all_homdensities),   # Shape (num_graphs, len(ks)).
        hom_counts=np.vstack(all_homcounts),    # Shape (num_graphs, len(ks)).
        pattern_sizes=pattern_sizes,
        pattern_edges=pattern_edges,
        graph_sizes=graph_sizes,
    )
