import argparse
from collections import Counter

import numpy as np
import pickle
import warnings

from matplotlib import pyplot as plt

from utils import normalize_counts, read_count_data, density_distance

warnings.filterwarnings("ignore", category=DeprecationWarning)  # np.abs complains otherwise.


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str,
                        help="distance metric used to compute the distance between count/density vectors",
                        default="l_inf")
    parser.add_argument("--data_path", default="old_maybe_wrong/OGBG-MOLHIV_full_kernel_16_50_run")
    parser.add_argument("--run", type=int, default=8)
    args = parser.parse_args()

    counts, graph_sizes, pattern_sizes, dataset_length = read_count_data(f'{args.data_path}{args.run}',
                                                                         dataset='molhiv', load_noisy=True)
    # I swear this is a nightmare.
    dataset_length = dataset_length//2

    # Compute densities and split between original and noisy graphs.
    densities = normalize_counts(counts, graph_sizes, pattern_sizes)
    densities_original = densities[:dataset_length]
    densities_noisy_graphs = densities[dataset_length:]

    distances, argmax_pattern = density_distance(densities_original, densities_noisy_graphs, args.metric)

    # TODO: from which pattern is the max achieved?
    # IF it happens that for many different patterns the count->density is the same
    # THEN something sus is happening maybe?

    # Look at the graphs which result in larger distances: do they have something specific?
    # How many graphs have a tiny distance? How do they look like?

    # Trying with another dataset.


    # Theoretical bound.
    with (open(f'data/{args.data_path}{args.run}.patterns', 'rb') as f):
        patterns = pickle.load(f)
        max_pattern_edges = max([g.number_of_edges() for g in patterns])
        max_pattern_nodes = min([g.number_of_nodes() for g in patterns])
        pattern_edges = [g.number_of_edges() for g in patterns]

    theoretical_bounds = ((np.ones(dataset_length) / graph_sizes[:dataset_length])**2) * max_pattern_edges * 2 * np.power(np.full(dataset_length, 4) / graph_sizes[:dataset_length], max_pattern_nodes - 2)
    #theoretical_bounds_v2 =

    with open(f'results/bound_vs_empirical/{args.data_path}_{args.run}.csv', 'w') as f:
        f.write('empirical,bound\n')
        for element in list(zip(distances, theoretical_bounds)):
            f.write(','.join(map(str, element)))
            f.write('\n')

    plt.title("Theoretical vs empirical bound")
    plt.xlabel("Theoretical bound")
    plt.ylabel("Empirical bound")
    plt.scatter(theoretical_bounds, distances, s=2)
    plt.show()

    threshold = 0.002
    small_distances = [d for d in distances if d<threshold]
    print(f'Number of distances smaller than {threshold}: {len(small_distances)}/{len(distances)}'
          f' = {len(small_distances)/len(distances):.2f}.')

    # Analysis of big and smalldistances.
    big_threshold=0.1
    big_idx = [i for i, d in enumerate(distances) if d > big_threshold]
    print(f"Printing info for distances larger than {big_threshold}.")
    if argmax_pattern is not None:
        print(f"{'Graph':<10}{'Graph size':<15}{'Pattern':<10}{'Pattern size':<15}{'Counts':<15}"
              f"{'NGraph':<10}{'NGraph size':<15}{'NPattern':<10}{'NPattern size':<15}{'NCounts':<10}"
              f"{'Distance':<15}")
        for i in big_idx:
            j = i+dataset_length
            print(f"{i:<10}"
                  f"{graph_sizes[i]:<15}"
                  f"{argmax_pattern[i]:<10}"
                  f"{pattern_sizes[argmax_pattern[i]]:<15}"
                  f"{counts[i][argmax_pattern[i]]:<15}"
                  f"{j:<10}"
                  f"{graph_sizes[j]:<15}"
                  f"{argmax_pattern[i]:<10}"
                  f"{pattern_sizes[argmax_pattern[i]]:<15}"
                  f"{counts[j][argmax_pattern[i]]:<10}"
                  f"{distances[i]:<10.2f}")


    small_threshod = 0.0001
    small_idx = [i for i, d in enumerate(distances) if d < small_threshod]
    print(f"\n\nPrinting info for smaller than {small_threshod}.")
    if argmax_pattern is not None:
        print(f"{'Graph':<10}{'Graph size':<15}{'Pattern':<10}{'Pattern size':<15}{'Counts':<15}"
              f"{'NGraph':<10}{'NGraph size':<15}{'NPattern':<10}{'NPattern size':<15}{'NCounts':<10}"
              f"{'Distance':<15}")
        for i in small_idx:
            j = i+dataset_length
            print(f"{i:<10}"
                  f"{graph_sizes[i]:<15}"
                  f"{argmax_pattern[i]:<10}"
                  f"{pattern_sizes[argmax_pattern[i]]:<15}"
                  f"{counts[i][argmax_pattern[i]]:<15}"
                  f"{j:<10}"
                  f"{graph_sizes[j]:<15}"
                  f"{argmax_pattern[i]:<10}"
                  f"{pattern_sizes[argmax_pattern[i]]:<15}"
                  f"{counts[j][argmax_pattern[i]]:<10}"
                  f"{distances[i]:<10.8f}")

    plt.title("Graphs size vs Distances")
    plt.xlabel("Graph size")
    plt.ylabel("Distance")
    plt.scatter(graph_sizes[:dataset_length], distances)
    plt.show()

    most_common_pattern = Counter(argmax_pattern).most_common(1)[0][0]
    plt.title(f"Graphs size vs counts for pattern {most_common_pattern} of size {pattern_sizes[most_common_pattern]}")
    plt.xlabel("Graph size")
    plt.ylabel("Counts")
    plt.scatter(graph_sizes[:dataset_length], counts[:dataset_length, most_common_pattern])
    plt.show()


if __name__ == '__main__':
    main()

