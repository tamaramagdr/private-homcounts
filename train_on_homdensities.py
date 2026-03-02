import argparse
import pickle

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR, SVC

from ogbg_cycles import get_ogbg_cycle_hom_features
from noise_and_sensitivity import smooth_sensitivities_tCDP, global_sensitivity
from sbm import build_sbm_cycles_dataset, compute_tree_features_alt
from tCDP import compute_epsilon_from_rho_delta
from train_aux import train_on_molhiv, train_on_molbace, train_on_molhiv_with_features, train_on_sbm, \
    train_on_molbace_with_features, train_on_molbbbp, train_on_molbbbp_with_features, \
    train_on_mollipo, train_on_mollipo_with_features, train_on_reddit_binary, \
    train_on_reddit_multi, train_on_github
from utils import normalize_counts, read_count_data


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--noise", type=float, default=1e-10)
    parser.add_argument("--rho", type=float, default=0.01)
    # parser.add_argument("--rho", type=float, default="inf")
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--dataset", default="molhiv")
    # parser.add_argument("--dataset", default="github_stargazers")
    # parser.add_argument("--dataset", default="reddit-multi-5k")
    parser.add_argument("--hom_size", type=int, default=16)
    # parser.add_argument("--pattern_count", type=int, default=50)
    # parser.add_argument("--pattern_count_ablation", type=int, default=50)
    parser.add_argument("--pattern_count", type=int, default=50)
    parser.add_argument("--pattern_count_ablation", type=int, default=50)
    parser.add_argument("--max_treewidth", type=int, default=1)
    parser.add_argument("--data_root", default="/homomorphism_counts")
    parser.add_argument("--save_path", default="test")
    parser.add_argument("--run", type=int, default=69)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--use_features", action="store_true", default=False)
    parser.add_argument("--features_only", action="store_true", default=False)
    parser.add_argument("--save_material", action="store_true", default=False)
    parser.add_argument("--run_attacks", action="store_true", default=False)
    parser.add_argument("--use_cycles_and_tw", action="store_true", default=False)
    parser.add_argument("--global_sensitivity_ablation", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Only used to avoid some computation if epsilon is infinite.
    epsilon = args.rho

    if args.rho == "inf":
        args.rho = np.inf

    data_path = f"{args.data_root}/{args.dataset.upper()}_{args.hom_size}_{args.pattern_count}_s{args.run}_tw{args.max_treewidth}"

    pickle_me = True
    if "cycle" in args.dataset:
        k_max = 10
        # Remove the "cycle_" prefix.
        dataset = args.dataset[6:]
        data = get_ogbg_cycle_hom_features(k_max, dataset=dataset)
        counts = torch.tensor(data["hom_counts"])
        pattern_sizes = data["pattern_sizes"]
        pattern_edges = data["pattern_edges"]
        dataset_length = len(data["hom_counts"])
        graph_sizes = data["graph_sizes"]
        if args.use_cycles_and_tw:
            print("Using both cycle and treewidth counts.")
            data_path = f"{args.data_root}/{dataset.upper()}_{args.hom_size}_{args.pattern_count}_s{args.run}_tw{args.max_treewidth}"
            tw_counts, tw_graph_sizes, tw_pattern_sizes, dataset_length = read_count_data(data_path, dataset, pickle_me=pickle_me)
            # Stack cycle counts with treewidth counts.
            counts = torch.cat([counts, tw_counts], dim=1)
            # Concatenate pattern sizes and edges.
            pattern_sizes = torch.cat([torch.tensor(pattern_sizes), torch.tensor(tw_pattern_sizes)], dim=0).tolist()
            with open(f'data/{data_path}.patterns', 'rb') as f:
                patterns = pickle.load(f)
                tw_pattern_edges = [g.number_of_edges() for g in patterns]
                # Handle the case where we padded patterns.
                if len(tw_pattern_edges) > len(tw_pattern_sizes):
                    tw_pattern_edges = tw_pattern_edges[-len(tw_pattern_sizes):]
            pattern_edges = torch.cat([torch.tensor(pattern_edges), torch.tensor(tw_pattern_edges)], dim=0).tolist()
    elif args.dataset == "sbm" or args.dataset == "tree_sbm":
        nodes_in_sbm = 200
        data = build_sbm_cycles_dataset(n=nodes_in_sbm, pbar=0.5, deltas=(0.08,0.16,0.24,0.32), k_max=4,
                                        graphs_per_class=100, seed=args.seed)
        counts = torch.tensor(data["hom_counts"])
        pattern_sizes = data["pattern_sizes"]
        pattern_edges = data["pattern_edges"]
        labels = data["labels"]
        dataset_length = len(counts)
        graph_sizes = [nodes_in_sbm]*dataset_length

        # If instead we want trees.
        if args.dataset == "tree_sbm":
            with open(f'data/{data_path}.patterns', 'rb') as f:
                patterns = pickle.load(f)
                pattern_edges = [g.number_of_edges() for g in patterns]
                tree_out = compute_tree_features_alt(data['graphs'], patterns)
                counts = torch.tensor(tree_out["hom_counts"])
                pattern_sizes = tree_out["pattern_sizes"]
    else:
        counts, graph_sizes, pattern_sizes, dataset_length = read_count_data(data_path, args.dataset, pickle_me=pickle_me)


    if args.pattern_count_ablation < args.pattern_count:
        print(f"Using only the first {args.pattern_count_ablation} patterns.")
        counts = counts[:, :args.pattern_count_ablation]
        pattern_sizes = pattern_sizes[:args.pattern_count_ablation]

    densities = normalize_counts(counts, graph_sizes, pattern_sizes)
    # This handles the (deprecated, but just in case) case where the dataset is padded with other graphs.
    # It should do nothing, in most cases.
    densities_original = densities[:dataset_length]
    if args.save_material:
        torch.save(densities, f'{args.dataset}.densities')

    # Get theoretical bound for each pattern.
    if "sbm" not in args.dataset and "cycle" not in args.dataset: # and not skip_pattern_edges:
        with open(f'data/{data_path}.patterns', 'rb') as f:
            patterns = pickle.load(f)
            pattern_edges = [g.number_of_edges() for g in patterns]
            # Handle the case where we padded patterns.
            if len(pattern_edges) > len(pattern_sizes):
                pattern_edges = pattern_edges[-len(pattern_sizes):]

    def top_1_attack(densities, densities_original):
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        nn.fit(densities_original)
        distances, indices = nn.kneighbors(densities)
        correct_assignments = np.sum(np.arange(len(densities)) == indices.flatten())
        accuracy = correct_assignments / len(densities)
        print(f"Top-1 hit accuracy: {accuracy:.4f}")
        return accuracy

    def top_k_hits(densities, densities_original, k=10):
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(densities_original)
        distances, indices = nn.kneighbors(densities)
        hits = np.sum([i in indices[idx] for idx, i in enumerate(np.arange(len(densities)))])
        top_k_accuracy = hits / len(densities)
        print(f"Top-{k} hit accuracy: {top_k_accuracy:.4f}")
        return top_k_accuracy

    # Smooth sensitivity computation.
    max_degree = 6
    if "molhiv" in args.dataset:
        max_degree = 10
    # The SBM may have a significantly larger max degree.
    if "sbm" in args.dataset:
        "Chernoff bound for max degree in sbm with p=0.5, with probability 1-1/nodes_in_sbm"
        max_degree = 0.5 * nodes_in_sbm + np.sqrt((nodes_in_sbm - 1) * np.log(nodes_in_sbm))
        assert max_degree > 6
    if "reddit" in args.dataset or "github" in args.dataset:
        max_degree = -1  # No degree bound, the smooth sensitivity code handles this case.
    l_sens, l_vars = smooth_sensitivities_tCDP(pattern_edges=pattern_edges, graph_sizes=graph_sizes,
                                               epsilon=epsilon, pattern_sizes=pattern_sizes,
                                               delta=args.delta, degree=max_degree, rho=args.rho)

    # Sensitivity computed for each graph using the L2 norm of the local sensitivities.
    l_stds = np.sqrt(l_vars)
    graph_wise_l2_sensitivities = np.linalg.norm(l_sens, axis=1)
    beta = args.rho/5
    rho = args.rho
    # From the sensitivities, get the sigmas for the Gaussian noise.
    sigmas = graph_wise_l2_sensitivities / np.sqrt(2*rho)  # If computing guarantees.
    sigma_tensor = torch.tensor(sigmas).view(len(densities_original), 1)  # shape (n_graphs,1).
    # Very elegant handling of inf epsilon.
    if args.rho == np.inf:
        noise = np.zeros_like(l_stds)
        vectorized_local_densities = densities_original
    else:
        # Go properties of the variance and broadcasting!
        noise = torch.randn((len(densities_original), len(pattern_sizes))) * sigma_tensor
        vectorized_local_densities = densities_original + noise.numpy()

    # Definition of rho prime for tCDP guarantees, and conversion to (eps, delta)-DP guarantees.
    rho_prime = 2*rho + 4*len(pattern_sizes)*(beta**2)
    omega = 1/(4*beta)
    if args.rho == np.inf:
        eps_dp_guarantee = np.inf
    else:
        eps_dp_guarantee = compute_epsilon_from_rho_delta(rho_prime, omega, args.delta)
    print(f"Individual zCDP privacy guarantee: eps={eps_dp_guarantee:4f}, delta={args.delta:.2e}\n")

    if args.global_sensitivity_ablation:
        # Assuming patterns have m=16 nodes and thus m-1 edges. This is true in all the experiments.
        g_sensitivity, g_variance = global_sensitivity(15, graph_sizes, epsilon=1, delta=args.delta,
                                                       max_degree=max_degree)
        g_sigma = 2* g_sensitivity * np.log(1.25/args.delta)/1**2
        noise = np.random.randn(*densities_original.shape) * g_sigma
        vectorized_local_densities = densities_original + noise

    # Graph reconstruction attacks.
    top_1_accuracy = 0
    top_k_accuracy = 0
    if args.run_attacks:
        print("Running graph reconstruction attacks...")
        nonnandensities = np.nan_to_num(vectorized_local_densities, nan=0.0, posinf=0.0, neginf=0.0)
        top_1_accuracy = top_1_attack(nonnandensities, densities_original)
        top_k_accuracy = top_k_hits(nonnandensities, densities_original, k=10)

    # Training.
    if "molhiv" in args.dataset:
        if not args.use_features:
            print("Not using features.")
            acc, roc_auc, split_idx, y_train = train_on_molhiv(vectorized_local_densities, densities_original,
                                                               KNeighborsClassifier, n_neighbors=1000, weights='uniform', algorithm='auto')
        else:
            acc, roc_auc, split_idx, y_train = train_on_molhiv_with_features(vectorized_local_densities, densities_original,
                                                                             learning_model=KNeighborsClassifier,
                                                                             fetures_only=args.features_only, n_neighbors=1000, weights='uniform', algorithm='auto')
        print(f"KNN on {args.dataset}, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif "molbace" in args.dataset:
        if not args.use_features:
            acc, roc_auc, split_idx, y_train = train_on_molbace(vectorized_local_densities,
                                                                densities_original,
                                                                features_only=args.features_only,
                                                                use_features=args.use_features,
                                                                learning_model=KNeighborsClassifier,
                                                                n_neighbors=120, weights='uniform', algorithm='auto')
        else:
            acc, roc_auc, split_idx, y_train = train_on_molbace_with_features(vectorized_local_densities,
                                                            densities_original,
                                                            features_only=args.features_only,
                                                            use_features=args.use_features,
                                                            learning_model=KNeighborsClassifier,
                                                            n_neighbors=100, weights='uniform', algorithm='auto')
        print(f"KNN on {args.dataset}, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif "sbm" in args.dataset:
        acc, roc_auc = train_on_sbm(vectorized_local_densities, labels, KNeighborsClassifier,
                                    n_neighbors=5, weights='uniform', algorithm='auto')
        print(f"KNN on {args.dataset}, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif "molbbbp" in args.dataset:
        if not args.use_features:
            acc, roc_auc, split_idx, y_train = train_on_molbbbp(vectorized_local_densities, densities_original,
                                                                learning_model=RandomForestClassifier, n_estimators=200, min_samples_split=10, max_depth=None, random_state=args.seed)
        else:
            acc, roc_auc, split_idx, y_train = train_on_molbbbp_with_features(vectorized_local_densities, densities_original,
                                                                              learning_model=RandomForestClassifier,
                                                                              fetures_only=args.features_only,
                                                                               n_estimators=200, min_samples_split=10, max_depth=None, random_state=args.seed)
        print(f"KNN on {args.dataset}, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif "mollipo" in args.dataset:
        if not args.use_features:
            acc, roc_auc, split_idx, y_train = train_on_mollipo(vectorized_local_densities, densities_original,
                                                                learning_model=SVR, kernel='linear', C=1.0, epsilon=0.2, gamma='scale')
        else:
            acc, roc_auc, split_idx, y_train = train_on_mollipo_with_features(vectorized_local_densities, densities_original,
                                                                              learning_model=SVR,
                                                                              fetures_only=args.features_only,
                                                                              kernel='linear', C=1.0, epsilon=0.2, gamma='scale')
        print(f"KNN on {args.dataset}, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif "reddit-binary" in args.dataset:
        if isinstance(vectorized_local_densities, torch.Tensor):
            vectorized_local_densities = vectorized_local_densities.numpy()
        if isinstance(densities_original, torch.Tensor):
            densities_original = densities_original.numpy()
        graph_sizes_col = np.array(graph_sizes).reshape(-1, 1)
        vectorized_local_densities = np.hstack([vectorized_local_densities, graph_sizes_col])
        densities_original = np.hstack([densities_original, graph_sizes_col])
        if not args.use_features:
            print("Not using features.")
            acc, roc_auc, split_idx, y_train = train_on_reddit_binary(vectorized_local_densities, densities_original,
                                                                    KNeighborsClassifier, n_neighbors=300, weights='uniform', algorithm='auto')
        else:
            pass
        print(f"KNN on {args.dataset}, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif "github" in args.dataset:
        if isinstance(vectorized_local_densities, torch.Tensor):
            vectorized_local_densities = vectorized_local_densities.numpy()
        if isinstance(densities_original, torch.Tensor):
            densities_original = densities_original.numpy()
        graph_sizes_col = np.array(graph_sizes).reshape(-1, 1)
        vectorized_local_densities = np.hstack([vectorized_local_densities, graph_sizes_col])
        densities_original = np.hstack([densities_original, graph_sizes_col])
        if not args.use_features:
            print("Not using features.")
            acc, roc_auc, split_idx, y_train = train_on_github(vectorized_local_densities, densities_original,
            learning_model=RandomForestClassifier, n_estimators=50, min_samples_split=10, max_depth=None, random_state=args.seed)
        else:
            pass
        print(f"KNN on {args.dataset}, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif "reddit-multi" in args.dataset:
        if isinstance(vectorized_local_densities, torch.Tensor):
            vectorized_local_densities = vectorized_local_densities.numpy()
        if isinstance(densities_original, torch.Tensor):
            densities_original = densities_original.numpy()
        graph_sizes_col = np.array(graph_sizes).reshape(-1, 1)
        vectorized_local_densities = np.hstack([vectorized_local_densities, graph_sizes_col])
        densities_original = np.hstack([densities_original, graph_sizes_col])
        if not args.use_features:
            print("Not using features.")
            acc, roc_auc, split_idx, y_train = train_on_reddit_multi(vectorized_local_densities, densities_original,
                                                                     learning_model=RandomForestClassifier, n_estimators=200, min_samples_split=10, max_depth=None, random_state=args.seed)
        else:
            pass
        print(f"KNN on {args.dataset}, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")

    # Writing results to file.
    output_file = f"{args.output_file}_{args.dataset.upper()}_{args.hom_size}_" \
    f"{args.pattern_count_ablation}_s{args.run}_tw{args.max_treewidth}" \
    f".csv" if hasattr(args, 'output_file') and args.output_file else \
    f'results/classifier_on_homdensities/{args.save_path}_{args.dataset.upper()}_{args.hom_size}_' \
    f'{args.pattern_count}_s{args.run}_tw{args.max_treewidth}_rho{args.rho}_' \
    f'delta{args.delta}_seed{args.seed}.csv'

    # Handle deprecated value.
    sep=0

    with open(output_file, 'a') as f:
        f.write(f"{args.run},{args.noise},{args.rho},{args.delta},{roc_auc:.4f},{acc:.4f},{args.seed},{sep:.4f},{sep},"
                f"{top_1_accuracy:.5f},{top_k_accuracy},{eps_dp_guarantee},{args.max_treewidth}\n")
        f.flush()

if __name__ == '__main__':
    main()

