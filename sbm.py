import networkx as nx
import numpy as np
from collections import deque

def sample_balanced_2block_sbm(n, p_in, p_out, rng):
    """Sample adjacency matrix of a balanced 2-block SBM."""
    n1 = n // 2 # Technically unbalanced if n is odd, but close enough.
    prob_matrix = np.empty((n, n))
    # Set up probability matrix.
    prob_matrix[:n1, :n1] = p_in
    prob_matrix[n1:, n1:] = p_in
    prob_matrix[:n1, n1:] = p_out
    prob_matrix[n1:, :n1] = p_out
    # Sample upper triangle and make symmetric.
    U = rng.uniform(size=(n, n))
    A = (U < prob_matrix).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    return A

def cycle_hom_densities_and_counts(A, ks):
    """For cycles, homdensities can be computed via traces of powers of A.
    For a cycle C_k, hom(C_k,G) = tr(A^k) and t(C_k,G) = tr(A^k)/n^k.
    """
    n = A.shape[0]
    ks = sorted(list(ks))
    max_k = ks[-1]
    Apow = A.copy()
    cur_k = 1
    traces = {1: np.trace(Apow)}
    while cur_k < max_k:
        Apow = Apow @ A
        cur_k += 1
        traces[cur_k] = np.trace(Apow)
    densities = []
    counts = []
    for k in ks:
        tr = traces[k]
        counts.append(tr)
        densities.append(tr / (n ** k))
    return np.array(densities), np.array(counts)

def build_sbm_cycles_dataset(n=100, pbar=0.5, deltas=(0.1, 0.15), k_max=4, graphs_per_class=5, seed=0):
    """Return dict with SBM graphs, hom densities, counts, and pattern metadata."""
    rng = np.random.default_rng(seed)
    ks = list(range(3, k_max + 1))
    pattern_sizes = np.array(ks, dtype=int)   # |V(C_k)| = k
    pattern_edges = np.array(ks, dtype=int)   # |E(C_k)| = k

    all_dens = []
    all_counts = []
    labels = []
    graphs = []
    for c, delta in enumerate(deltas):
        p_in, p_out = pbar + delta, pbar - delta
        for _ in range(graphs_per_class):
            A = sample_balanced_2block_sbm(n, p_in, p_out, rng)
            dens, counts = cycle_hom_densities_and_counts(A, ks)
            all_dens.append(dens)
            all_counts.append(counts)
            labels.append(c)
            graphs.append(A)
    return dict(
        graphs=np.array(graphs, dtype=object),
        hom_densities=np.vstack(all_dens),   # Shape (num_graphs, len(ks)).
        hom_counts=np.vstack(all_counts),    # Shape (num_graphs, len(ks)).
        labels=np.array(labels),
        pattern_sizes=pattern_sizes,
        pattern_edges=pattern_edges,
        n=np.array([n]),
        ks=np.array(ks),
        deltas=np.array(deltas),
        pbar=np.array([pbar]),
    )


def compute_tree_features_alt(graphs, patterns):
    """
    Compute homomorphism counts and densities for tree patterns on given graphs.

    Parameters
    ----------
    graphs : list of np.ndarray
        List of adjacency matrices (n x n) for the host graphs.
    patterns : list of nx.Graph
        List of pattern graphs (each must be a tree).

    Returns
    -------
    dict
        {
            "hom_counts": np.ndarray (num_graphs x num_patterns),
            "hom_densities": np.ndarray (num_graphs x num_patterns),
            "pattern_sizes": np.ndarray (num_patterns,),
            "pattern_edges": np.ndarray (num_patterns,),
            "pattern_names": np.ndarray (num_patterns,),
        }
    """

    num_graphs = len(graphs)
    num_patterns = len(patterns)

    pattern_names = [f"Tree(p={p.number_of_nodes()})" for p in patterns]
    pattern_num_vertices = [p.number_of_nodes() for p in patterns]
    pattern_num_edges = [p.number_of_edges() for p in patterns]

    homomorphism_counts = np.zeros((num_graphs, num_patterns), dtype=float)
    homomorphism_densities = np.zeros((num_graphs, num_patterns), dtype=float)

    for graph_idx, A in enumerate(graphs):
        if graph_idx % 100 == 0:
            print(f"processing graph {graph_idx+1} of {num_graphs}")
        # n = A.shape[0]
        for pattern_idx, pattern in enumerate(patterns):
            # Run tree DP.
            root = list(pattern.nodes)[0]  # Arbitrary root.
            count, density = _tree_hom(A, pattern, root=root)
            homomorphism_counts[graph_idx, pattern_idx] = count
            homomorphism_densities[graph_idx, pattern_idx] = density

    return dict(
        hom_counts=homomorphism_counts,
        hom_densities=homomorphism_densities,
        pattern_sizes=np.array(pattern_num_vertices, dtype=int),
        pattern_edges=np.array(pattern_num_edges, dtype=int),
        pattern_names=np.array(pattern_names, dtype=object),
    )

def _tree_hom(A, pattern: nx.Graph, root=0):
    """Helper: count homomorphisms from tree pattern into host graph."""
    n = A.shape[0]
    host_neighbors = [np.where(A[i] > 0)[0] for i in range(n)]

    # Root the pattern.
    parent = {root: None}
    order = []
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in pattern.neighbors(u):
            if v not in parent:
                parent[v] = u
                stack.append(v)

    # Bottom-up DP.
    DP = {u: np.ones(n, dtype=np.int64) for u in pattern.nodes}
    for u in reversed(order):
        for x in range(n):
            prod = 1
            for v in pattern.neighbors(u):
                if parent.get(v) == u:  # child
                    s = DP[v][host_neighbors[x]].sum()
                    prod *= s
            DP[u][x] = prod

    hom_count = float(DP[root].sum())
    hom_density = hom_count / (n ** pattern.number_of_nodes())
    return hom_count, hom_density


# Alternative pattern computations to be computed.
def path_hom_formula(A, ell):
    """
    Homomorphism count for a path of length ell using matrix powers.
    """
    n = A.shape[0]
    ones = np.ones((n, 1), dtype=int)
    A_power = np.linalg.matrix_power(A, ell)
    count = int((ones.T @ A_power @ ones)[0, 0])
    density = count / (n ** (ell + 1))
    return count, density

def star_hom_formula(A, r):
    """
    Homomorphism count for a star with r leaves using degree powers.
    """
    n = A.shape[0]
    deg = A.sum(axis=1)
    count = int(np.sum(deg ** r))
    density = count / (n ** (r + 1))
    return count, density


if __name__ == "__main__":
    # Sanity checking dynamic programming tree hom counts against formulas for stars and paths.
    A1 = np.array([[0,1,1],
                   [1,0,1],
                   [1,1,0]])
    A2 = np.array([[0,1,0],
                   [1,0,1],
                   [0,1,0]])
    graphs = [("Triangle K3", A1), ("Path P3", A2)]

    patterns = [
        ("Path(ell=1)", nx.path_graph(2)),
        ("Path(ell=2)", nx.path_graph(3)),
        ("Path(ell=3)", nx.path_graph(4)),
        ("Star(r=2)", nx.star_graph(2)),
        ("Star(r=3)", nx.star_graph(3)),
    ]

    dp_count = compute_tree_features_alt([A1, A2],
                                         [nx.path_graph(2), nx.path_graph(3), nx.path_graph(4),
                                          nx.star_graph(2), nx.star_graph(3)])['hom_counts']

    print(dp_count)
    for gname, A in graphs:
        print(f"\nHost graph: {gname}")
        for pname, P in patterns:
            if pname.startswith("Path"):
                ell = int(pname.split("=")[1][:-1])
                formula_count, formula_density = path_hom_formula(A, ell)
            else:
                r = int(pname.split("=")[1][:-1])
                formula_count, formula_density = star_hom_formula(A, r)
            print(f"{pname:12s} | Formula: {formula_count}")
