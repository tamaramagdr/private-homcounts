import numpy as np
from tqdm import tqdm

def global_sensitivity(max_pattern_edges, graph_sizes, epsilon, delta, max_degree=None):
    if max_degree is not None:
        max_sensitivities = []
        for patter_nodes in range(2, max_pattern_edges+1):
            sensitivities = [(2*(patter_nodes-1)/(graph_sizes[i]**2))*((max_degree/graph_sizes[i])**(patter_nodes-2)) for i in range(len(graph_sizes))]
            max_sensitivity = max(sensitivities)
            max_sensitivities.append(max_sensitivity)
        sensitivity = max(max_sensitivities)
    else:
        sensitivities = [2*(max_pattern_edges)/(graph_sizes[i]**2) for i in range(len(graph_sizes))]
        sensitivity = max(sensitivities)
    variance = 2 * (sensitivity * np.log(1.25 / delta)) / (epsilon**2)
    return sensitivity, variance


def smooth_sensitivities_tCDP(pattern_edges, graph_sizes, pattern_sizes, epsilon, delta, rho,
                                                           degree=6):
    """
    Computes the smooth sensitivities for a set of patterns and graphs.
    Returns the sensitivities and variances for each pattern in each graph.
    The variances are already computed as per the tCDP definition, each with a budget of rho.
    """
    k_max = 6
    beta = rho/5
    graph_sensitivities = []

    # If epsilon is inf, just return matrices of zeros.
    if epsilon == np.inf:
        graph_sensitivities = np.zeros((len(graph_sizes), len(pattern_edges)))
        variances = np.zeros((len(graph_sizes), len(pattern_edges)))
        return graph_sensitivities, variances

    for i in tqdm(range(len(graph_sizes)), desc="Computing smooth sensitivities..."):
        # For each graph, compute the vector of sensitivities.
        local_sensitivities = []
        for j in range(len(pattern_edges)):
            patter_graph_local_sensitivity = compute_local_bounded_degree_sensitivity(graph_sizes[i], pattern_edges[j],
                                                                                      pattern_sizes[j],
                                                                                      degree, edge_distance=1)

            # Fixed local density computation, independent of k.
            k_candidates = [patter_graph_local_sensitivity*1 for k in range(1, k_max+1)]
            # Smooth sensitivity computation, depends on beta.
            max_candidate = smooth_sensitivity_aux(k_candidates, beta, k_max)
            local_sensitivities.append(max_candidate)
        graph_sensitivities.append(local_sensitivities)

    graph_sensitivities = np.array(graph_sensitivities)

    # Variance computation, depends on rho.
    variances = (graph_sensitivities**2) / (2*rho)

    return graph_sensitivities, variances


def smooth_sensitivity_aux(local_sensitivities, beta, k_max=4):
    smooth_sensitivities = [np.exp(-beta * k) * local_sensitivities[k] for k in range(k_max)]
    return max(smooth_sensitivities)

def compute_local_bounded_degree_sensitivity(n_G, e_F, n_F, delta_max, edge_distance=1):
    # if delta_max == -1:  # Use this to always use delta_max.
    if delta_max == -1 or delta_max > n_G:
        delta_max = n_G
    return ( (e_F/(n_G**2)) * 2 * ( (delta_max/n_G)**(n_F-2) ) )*edge_distance
