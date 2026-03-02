import json
import pickle

import numpy as np
import torch
from scipy.stats import wasserstein_distance


def normalize_densities_tensor(densities):
    densities_min = torch.min(densities, dim=1, keepdim=True).values
    densities_max = torch.max(densities, dim=1, keepdim=True).values
    densities = (densities - densities_min) / (densities_max - densities_min)
    return densities


def normalize_counts(counts, graph_sizes, pattern_sizes):
    """
    Normalize the counts by the graph sizes and pattern sizes.
    """

    # Convert graph sizes and pattern sizes to tensors
    graph_sizes_tensor = torch.tensor(graph_sizes).float().unsqueeze(1)  # shape [len(graph_sizes), 1].
    pattern_sizes_tensor = torch.tensor(pattern_sizes).float().unsqueeze(0)  # shape [1, len(pattern_sizes)].
    # To get shape [len(graph_sizes), len(pattern_sizes)], broadcasting magic.
    normalization = torch.pow(graph_sizes_tensor, (-1.) * pattern_sizes_tensor)

    # Normalize counts
    densities = (torch.mul(counts, normalization))

    return densities


def density_distance(densities_noise, densities_original, metric):
    """
    Compute the distance between the noisy and original densities.
    Returns the distance and the index of the pattern that gives the maximum distance if l_inf is used.
    """
    if metric == "l_inf":
        # Should do what we want, getting the max pattern for each graph.
        l_abs_val = np.abs(densities_noise - densities_original)
        # Compute argmax to identify the patterns.
        argmax = np.argmax(l_abs_val, axis=1)
        return torch.max(l_abs_val, axis=1)[0].tolist(), argmax
    elif metric == "wasserstein":
        return ([wasserstein_distance(densities_noise[i], densities_original[i]) for i in range(len(densities_noise))],
                None)
    elif metric == "euclidean":
        return torch.sqrt(torch.sum((densities_noise - densities_original) ** 2, axis=1)).tolist(), None
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def read_count_data(data_path, dataset, load_noisy=False, pickle_me=True):
    """
    Read the data from the file and compute the densities.
    """
    if pickle_me is True:
        with open (f'data/{data_path}.hom', 'rb') as f:
            homcount_data = pickle.load(f)
    try:
        path = f'data/{data_path}.homson'
        with open(path, 'rb') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found: help me!")

    if pickle_me is True:
        dataset_length = len(homcount_data)
    else:
        dataset_length = len(data['data'])
    if dataset == 'molhiv':
        # TODO: Hacky way to re-use counts obtained from the noisy runs.
        dataset_length = 41127
        if load_noisy is True:
            dataset_length = dataset_length*2

    counts_list = []
    graph_sizes = []
    pattern_sizes = data['pattern_sizes']

    # Fix indices because of bug in previous code.
    for idx, i in enumerate(data['data']):
        i['idx'] = idx

    if load_noisy is True:
        for i in range(dataset_length):
            counts_list.append(homcount_data[i])
            graph_sizes.append(data['data'][i]['vertices'])
    else:
        for i in data['data'][:dataset_length]:
            if i['idx'] < dataset_length:
                counts_list.append(i['counts'])
            else:
                # This should make sure we don't read noisy counts.
                raise ValueError(f"Index {i['idx']} is out of bounds for dataset length {dataset_length}.")
                # counts_list.append(j)
            graph_sizes.append(i['vertices'])

    counts = torch.from_numpy(np.array(counts_list))

    return counts, graph_sizes, pattern_sizes, dataset_length