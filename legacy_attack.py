import argparse
import pickle

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVC, SVR

from noise_and_sensitivity import get_noise, compute_per_element_sensitivies, compute_per_element_local_sensitivities
from train_aux import train_on_zinc, train_on_molhiv, MLP, train_on_csl
from utils import normalize_counts, read_count_data, compute_interclass_separability


# from ogb.graphproppred import GraphPropPredDataset # , PygGraphPropPredDataset


def main():

    parser = argparse.ArgumentParser()

    # parser.add_argument("--noise", type=float, default=1e-10)
    parser.add_argument("--noise", type=float, default=1e-10)
    parser.add_argument("--epsilon", type=float, default=np.inf)
    parser.add_argument("--delta", type=float, default=10e-5)
    parser.add_argument("--dataset", default="molhiv")
    # parser.add_argument("--dataset", default="zinc")
    # parser.add_argument("--dataset", default="csl")
    parser.add_argument("--hom_size", type=int, default=16)
    parser.add_argument("--pattern_count", type=int, default=50)
    parser.add_argument("--max_treewidth", type=int, default=1)
    # parser.add_argument("--data_root", default="/homomorphism_counts")
    # parser.add_argument("--data_root", default="/old_maybe_wrong")
    parser.add_argument("--data_root", default="/maybe_noisye")
    parser.add_argument("--save_path", default="test")
    parser.add_argument("--run", type=int, default=69)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None)

    # this block is all copy-pasted from https://github.com/pwelke/homcount
    parser.add_argument("--grid_search", action="store_true", default=False)
    parser.add_argument("--C", type=float, help="SVC's C parameter.", default=1e4)
    parser.add_argument("--kernel", type=str, help="SVC kernel function.", default="rbf")
    parser.add_argument("--degree", type=int, help="Degree of `poly` kernel.", default=2)
    parser.add_argument("--gamma", type=float, help="SVC's gamma parameter.", default=40.0)
    parser.add_argument("--gs_nfolds", type=int, default=5)
    parser.add_argument("--scaler", type=str, default="standard",
                        help="Name of data scaler to use as the preprocessing step")

    args = parser.parse_args()

    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    if args.epsilon == "inf":
        args.epsilon = np.inf

    data_path = f"{args.data_root}/{args.dataset.upper()}_{args.hom_size}_{args.pattern_count}_s{args.run}_tw{args.max_treewidth}"
    data_path = f"{args.data_root}/OGBG-MOLHIV-noisy_full_kernel_16_50_run1"

    counts, graph_sizes, pattern_sizes, dataset_length = read_count_data(data_path, args.dataset, load_noisy=True)
    # Quick mafs.
    dataset_length = dataset_length//2

    # Compute densities and split between original and noisy graphs.
    densities = normalize_counts(counts, graph_sizes, pattern_sizes)
    densities_original = densities[:dataset_length]
    densities_original_edge_flip = densities[dataset_length:]

    # Get theoretical bound for each pattern.
    with open(f'data/{data_path}.patterns', 'rb') as f:
        patterns = pickle.load(f)
        pattern_edges = [g.number_of_edges() for g in patterns]
        # TODO: these seem to be in the correct order, but double check.

    sensitivities, variances = compute_per_element_sensitivies(pattern_edges, graph_sizes, args.epsilon, args.delta)
    # alt_sensitivities, alt_variances = compute_per_element_local_sensitivities(pattern_edges, graph_sizes,
    #                                                                            args.epsilon, args.delta)
    stds = [np.sqrt(variances[i]) for i in range(len(variances))]
    # print(f"sensitivities: {sensitivities}")
    # print(f"sensitivities: {[f'{sensitivity:.5f}' for sensitivity in sensitivities]}")
    # print(f"stds: {[f'{std:.5f}' for std in stds]}")

    # TODO: create many noisy embeddings for the same graphs, then use them as data to be trained.

    # noise = get_noise(counts, dataset_length, stds)
    # densities = densities_original + noise
    # densities_edge_flip = densities_original_edge_flip + noise


    def generate_noisy_embeddings_and_train_classifier(densities_original, densities_original_edge_flip,
                                                       counts, dataset_length, stds, num_noisy_samples=100,
                                                       subset_size=1000):
        X = []
        y = []

        for i in range(num_noisy_samples):
            print(f"Sample i: {i}")
            noise = get_noise(counts, dataset_length, stds)
            noisy_original = densities_original + noise
            noise = get_noise(counts, dataset_length, stds)
            noisy_edge_flip = densities_original_edge_flip + noise


            # Compare element-wise and determine majority
            original_larger_count = 0
            for original, edge_flip in zip(noisy_original, noisy_edge_flip):
                # Compare element-wise and count how many elements are larger
                original_is_larger = np.sum(original > edge_flip)
                edge_flip_is_larger = np.sum(edge_flip > original)

                # Determine which vector is larger based on majority
                if original_is_larger > edge_flip_is_larger:
                    original_larger_count += 1

            total_pairs = len(noisy_original)
            print(f"Original is larger in {original_larger_count}/{total_pairs}, for a fraction {original_larger_count / total_pairs}")

            # Take subset
            noisy_original = noisy_original[:subset_size]
            noisy_edge_flip = noisy_edge_flip[:subset_size]

            X.extend(noisy_original)
            y.extend([0] * len(noisy_original))  # Class 0 for original
            X.extend(noisy_edge_flip)
            y.extend([1] * len(noisy_edge_flip))  # Class 1 for edge-flip

        X = np.array(X)
        y = np.array(y)

        # Odd thing to make sure I get to not split individual graphs across training and test.
        X = np.array(X).reshape(-1, num_noisy_samples * 2, X[0].shape[-1])
        y = np.array(y).reshape(-1, num_noisy_samples * 2)

        X_train_chunks, X_test_chunks, y_train_chunks, y_test_chunks = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train = X_train_chunks.reshape(-1, X.shape[-1])
        y_train = y_train_chunks.flatten()
        X_test = X_test_chunks.reshape(-1, X.shape[-1])
        y_test = y_test_chunks.flatten()

        # This was the old version.
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto')
        # classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Classifier accuracy: {accuracy:.4f}")

        return classifier, accuracy



    classifier, attack_accuracy = generate_noisy_embeddings_and_train_classifier(densities_original,
                                                                          densities_original_edge_flip,
                                                                          counts, dataset_length, stds)


    # This is what a msc in data science gives you.
    # print(args.seed)
    output_file = f"{args.output_file}_knn_{args.dataset.upper()}_{args.hom_size}_" \
    f"{args.pattern_count}_s{args.run}_tw{args.max_treewidth}" \
    f".csv" if hasattr(args, 'output_file') and args.output_file else \
    f'results/classifier_on_homcounts/{args.save_path}_{args.dataset.upper()}_{args.hom_size}_' \
    f'{args.pattern_count}_s{args.run}_tw{args.max_treewidth}_eps{args.epsilon}_' \
    f'delta{args.delta}_seed{args.seed}.csv'


    with open(output_file, 'a') as f:
        f.write(f"{args.run},{args.noise},{args.epsilon},{args.delta},{args.seed},{np.max(sensitivities)},"
                f"{attack_accuracy:.4f}\n")
        f.flush()






if __name__ == '__main__':
    main()

