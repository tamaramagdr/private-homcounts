import numpy as np
import torch
from tqdm import tqdm


def get_noise_single(counts, dataset_length, sigma, pattern_edges=None, range=None, truncated=False):
    noise = torch.randn(counts[:dataset_length].shape)
    if pattern_edges is not None:
        noise.mul_(torch.tensor(pattern_edges).view(1, -1) * sigma)
    else:
        noise.mul_(torch.full(counts[:dataset_length].shape, sigma))
    if truncated:
        assert range is not None, "Range should not be None if truncated is True."
        noise = np.clip(noise, -range, range)
    return noise

def get_noise(counts, dataset_length, stds, range=None, truncated=False):
    noise = torch.normal(mean=0.0, std=torch.tensor(stds).view(1, -1).expand_as(counts[:dataset_length]))
    if truncated:
        assert range is not None, "Range should not be None if truncated is True."
        noise = np.clip(noise, -range, range)
    return noise

def get_matrix_noise(stds):
    stds_tensor = torch.tensor(stds)
    if torch.all(stds_tensor == 0):
        noise = torch.zeros_like(stds_tensor)
    else:
        noise = torch.normal(mean=0.0, std=stds_tensor)
    return noise

def compute_wrong_per_element_sensitivies(pattern_edges, graph_sizes, epsilon, delta):
    # TODO: this is bullshit.
    sensitivities = [pattern_edges[i] / graph_sizes[i]**2 for i in range(len(pattern_edges))]
    variances = [2 * (sensitivities[i] * np.log(1.25 / delta)) / (epsilon**2) for i in range(len(sensitivities))]
    return sensitivities, variances

def compute_per_element_sensitivies(pattern_edges, graph_sizes, epsilon, delta):
    # TODO: this is maybe not bullshit?
    # sensitivities = [pattern_edges[i] / graph_sizes[i]**2 for i in range(len(pattern_edges))]
    sensitivities = [max(pattern_edges[p] / graph_sizes[g]**2 for g in range(len(graph_sizes))) for p in range(len(pattern_edges))]
    variances = [2 * (sensitivities[i] * np.log(1.25 / delta)) / (epsilon**2) for i in range(len(sensitivities))]
    return sensitivities, variances

def compute_bounded_degree_per_element_sensitivities(pattern_edge, graph_sizes, epsilon, delta):
    sensitivities = []

def compute_per_element_local_sensitivities(pattern_edges, graph_sizes, epsilon, delta, tw=1, degree=5):
    sensitivities = [compute_local_density_sensitivity(graph_sizes[i], pattern_edges[i], tw, degree)
                     for i in range(len(pattern_edges))]
    assert 1==0, "TODO: here I may have forgotten a square"
    variances = [2 * (sensitivities[i] * np.log(1.25 / delta)) / (epsilon**2) for i in range(len(sensitivities))]
    return sensitivities, variances

def compute_local_density_sensitivity(n_G, e_F, tw=1, degree=5):
    return degree**(tw) * n_G**(tw - 1 - e_F)
    # TODO: really check this.

def compute_per_element_bounded_degree_local_sensitivities(pattern_edges, graph_sizes, pattern_sizes, epsilon, delta,
                                                           degree=6, vectorized=False):
    """
    Computes the local sensitivities for a set of patterns and graphs, using the default eps-delta guarantee.
    The other function with tCDP should allow for better utility tradeoff.
    """
    # TODO: double check that it's not garbage. Shouldn't be: https://dl.acm.org/doi/pdf/10.1145/1250790.1250803
    # Here epsilon and delta are the individual epsilon for each pattern, needs composition.
    # A(x) = f(x) + Z * S(x)/alpha, where S(x) is the beta-smooth upper bound to the local sensitivity of f(x),
    # and alpha = epsilon/sqrt(log(1/delta)), beta = epsilon/(2 * log(1/delta)).
    # Then with Z=1/2pi exp(-z^2/2) is (alpha,beta)-admissible.
    # beta = epsilon/(2 * np.log(1/delta))
    beta = delta
    assert beta <= epsilon/(2 * np.log(1/delta)), "Beta should be <= than epsilon/(2 * log(1/delta))"
    k_max = 4
    graph_sensitivities = []
    for i in tqdm(range(len(graph_sizes)), desc="Computing smooth sensitivities..."):
        # For each graph, compute the vector of sensitivities.
        local_sensitivities = []
        for j in range(len(pattern_edges)):
            patter_graph_local_sensitivity = compute_local_bounded_degree_sensitivity(graph_sizes[i], pattern_edges[j],
                                                                                      pattern_sizes[j],
                                                                                      degree, edge_distance=1)
            k_candidates = [patter_graph_local_sensitivity*k for k in range(1, k_max+1)]
            # Smooth sensitivity, depends on beta.
            max_candidate = smooth_sensitivity_aux(k_candidates, beta, k_max)
            local_sensitivities.append(max_candidate)
        graph_sensitivities.append(local_sensitivities)

    graph_sensitivities = np.array(graph_sensitivities)

    # Variance, depends on alpha, variance is that of (S(x)/alpha)^2.
    variances = 2 * (graph_sensitivities**2 * np.log(2 / delta)) / (epsilon**2)

    return graph_sensitivities, variances


def smooth_sensitivities_tCDP(pattern_edges, graph_sizes, pattern_sizes, epsilon, delta, rho,
                                                           degree=6):
    """
    Computes the smooth sensitivities for a set of patterns and graphs.
    Returns the sensitivities and variances for each pattern in each graph.
    The variances are already computed as per the tCDP definition, each with a budget of rho.
    """
    # TODO: double check this.
    k_max = 6
    # TODO: this is a uniform assignment that can likely be improved.
    beta = rho/5
    # beta = epsilon/(2 * np.log(1/delta))
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

            k_candidates = [patter_graph_local_sensitivity*k for k in range(1, k_max+1)]
            # Smooth sensitivity computation, depends on beta.
            max_candidate = smooth_sensitivity_aux(k_candidates, beta, k_max)
            local_sensitivities.append(max_candidate)
        graph_sensitivities.append(local_sensitivities)

    graph_sensitivities = np.array(graph_sensitivities)

    # Variance computation, depends on rho.
    variances = (graph_sensitivities**2) / (2*rho)

    return graph_sensitivities, variances

def zcdp(epsilon, delta, d):
    """
    epsilon and delta are the per-pattern parameters.
    """
    rho_per_pattern = (epsilon ** 2) / (2 * np.log(1 / delta))
    rho = d * rho_per_pattern
    return rho

def zcdp_to_dp(rho, target_delta):
    return rho + np.sqrt(4 * rho * np.log(1 / target_delta))

def smooth_sensitivity_aux(local_sensitivities, beta, k_max=4):
    smooth_sensitivities = [np.exp(-beta * k) * local_sensitivities[k] for k in range(k_max)]
    return max(smooth_sensitivities)

def compute_local_bounded_degree_sensitivity(n_G, e_F, n_F, delta_max, edge_distance=1):
    return ( (e_F/(n_G**2)) * 2 * ( (delta_max/n_G)**(n_F-2) ) )*edge_distance
