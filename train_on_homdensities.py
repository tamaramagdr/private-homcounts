import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from noise_and_sensitivity import get_noise, compute_per_element_sensitivies, compute_per_element_local_sensitivities, \
    compute_wrong_per_element_sensitivies, compute_per_element_bounded_degree_local_sensitivities, get_matrix_noise, \
    zcdp, zcdp_to_dp, smooth_sensitivities_tCDP
from tCDP import compute_epsilon_from_rho_delta, compute_epsilon_from_rho_delta
from train_aux import train_on_zinc, train_on_molhiv, MLP, train_on_csl, train_on_molbace, train_on_molpcba, \
    train_on_moltox21, train_on_molhiv_with_feature_perturbation, feature_zinc
from utils import normalize_counts, read_count_data, compute_interclass_separability, normalize_densities_tensor, \
    load_csl_data


# from ogb.graphproppred import GraphPropPredDataset # , PygGraphPropPredDataset


def main():

    parser = argparse.ArgumentParser()

    # parser.add_argument("--noise", type=float, default=1e-10)
    parser.add_argument("--noise", type=float, default=1e-10)
    # parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=1e-6)
    # parser.add_argument("--dataset", default="molhiv")
    parser.add_argument("--dataset", default="molbace")
    # parser.add_argument("--dataset", default="molpcba")
    # parser.add_argument("--dataset", default="moltox21")
    # parser.add_argument("--dataset", default="zinc")
    # parser.add_argument("--dataset", default="csl")
    parser.add_argument("--hom_size", type=int, default=16)
    parser.add_argument("--pattern_count", type=int, default=50)
    parser.add_argument("--max_treewidth", type=int, default=1)
    parser.add_argument("--data_root", default="/homomorphism_counts")
    parser.add_argument("--save_path", default="test")
    parser.add_argument("--run", type=int, default=69)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None)

    # this block is all copy-pasted from https://github.com/pwelke/homcount
    parser.add_argument("--grid_search", action="store_true", default=False)
    parser.add_argument("--C", type=float, help="SVC's C parameter.", default=1)
    parser.add_argument("--kernel", type=str, help="SVC kernel function.", default="rbf")
    parser.add_argument("--degree", type=int, help="Degree of `poly` kernel.", default=2)
    parser.add_argument("--gamma", type=float, help="SVC's gamma parameter.", default=4.0)
    parser.add_argument("--gs_nfolds", type=int, default=5)
    parser.add_argument("--scaler", type=str, default="standard",
                        help="Name of data scaler to use as the preprocessing step")

    args = parser.parse_args()

    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # TODO: as args.epsilon is not really used, this is ok, but should be made programmatic.
    args.epsilon = args.rho

    if args.rho == "inf":
        args.rho = np.inf

    data_path = f"{args.data_root}/{args.dataset.upper()}_{args.hom_size}_{args.pattern_count}_s{args.run}_tw{args.max_treewidth}"

    pickle_me = True
    if args.dataset == 'molbace':
        pickle_me = False
    counts, graph_sizes, pattern_sizes, dataset_length = read_count_data(data_path, args.dataset, pickle_me=pickle_me)

    # Compute densities and split between original and noisy graphs.
    densities = normalize_counts(counts, graph_sizes, pattern_sizes)
    densities_original = densities[:dataset_length]
    # densities_noisy_graphs = densities[dataset_length:]

    # Get theoretical bound for each pattern.
    with open(f'data/{data_path}.patterns', 'rb') as f:
        patterns = pickle.load(f)
        pattern_edges = [g.number_of_edges() for g in patterns]
        # TODO: these seem to be in the correct order, but double check.
    if len(pattern_sizes) != len(pattern_edges):
        new_pattern_edges = []
        j=0
        for i in range(len(pattern_edges)):
            if pattern_sizes[j] == patterns[i].number_of_nodes():
                new_pattern_edges.append(patterns[i].number_of_edges())
                j = j+1
        pattern_edges = new_pattern_edges


    def some_attack(densities, densities_original):
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        nn.fit(densities_original)
        distances, indices = nn.kneighbors(densities)
        correct_assignments = np.sum(np.arange(len(densities)) == indices.flatten())
        accuracy = correct_assignments / len(densities)
        print(f"Accuracy of some attack: {accuracy:.4f}")
        return accuracy

    def top_k_hits(densities, densities_original, k=10):
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(densities_original)
        distances, indices = nn.kneighbors(densities)
        hits = np.sum([i in indices[idx] for idx, i in enumerate(np.arange(len(densities)))])
        top_k_accuracy = hits / len(densities)
        print(f"Top-{k} hit accuracy: {top_k_accuracy:.4f}")
        return top_k_accuracy


    # GLOBAL.
    sensitivities, variances = compute_per_element_sensitivies(pattern_edges, graph_sizes, args.epsilon, args.delta)
    stds = [np.sqrt(variances[i]) for i in range(len(variances))]
    noise = get_noise(counts, dataset_length, stds)
    densities = densities_original + noise
    densities = normalize_densities_tensor(densities)
    print(f"density difference: {np.abs(densities_original-densities).sum():.3f}")
    # print(f"density difference avg: {(densities_original-densities).mean(axis=0)}")


    # LOCAL/SMOOTH.
    l_sens, l_vars = smooth_sensitivities_tCDP(pattern_edges=pattern_edges, graph_sizes=graph_sizes,
                                               epsilon=args.epsilon, pattern_sizes=pattern_sizes,
                                               delta=args.delta, degree=6, rho=args.rho)

    # Sensitivity computed for each PATTERN for each graph.
    # Composition is at worse n_patterns * rho budget for the whole graph.
    l_stds = np.sqrt(l_vars)
    # Very elegant handling of inf epsilon.
    if args.rho == np.inf:
        local_noise = np.zeros_like(l_stds)
    else:
        local_noise = get_matrix_noise(l_stds)
    local_densities = densities_original + local_noise
    local_densities = normalize_densities_tensor(local_densities)
    # overall_dp_guarantee = zcdp_to_dp(overall_zcdp_guarantee, target_delta)


    # Sensitivity computed for each graph as the L2 norm of the local sensitivities.
    # Should be nicer to compose, as it uses rho budget for the whole graph.
    graph_wise_l2_sensitivities = np.linalg.norm(l_sens, axis=1)
    beta = args.rho/5
    rho = args.rho
    # sigmas = graph_wise_l2_sensitivities / np.sqrt(2*(args.epsilon - 4*(beta**2))) # to get rho guaranteees.
    sigmas = graph_wise_l2_sensitivities / np.sqrt(2*rho)  # If computing guarantees.
    sigma_tensor = torch.tensor(sigmas).view(len(densities), 1)  # shape (n_graphs,1).
    # Very elegant handling of inf epsilon.
    if args.rho == np.inf:
        noise = np.zeros_like(l_stds)
        vectorized_local_densities = densities_original
    else:
        noise = torch.randn((len(densities), len(pattern_sizes))) * sigma_tensor  # Go go properties of the variance and broadcasting.
        vectorized_local_densities = densities_original + noise.numpy()
    vectorized_local_densities = normalize_densities_tensor(vectorized_local_densities)
    # dp_guarantee = compute_epsilon_from_rho_delta(args.epsilon, args.delta)
    # Actually trying not to make things up
    # dp_guarantee = compute_epsilon_from_rho_delta(2*rho + 4*len(pattern_sizes)*(beta**2), args.delta)
    rho_prime = 2*rho + 4*len(pattern_sizes)*(beta**2)
    omega = 1/(4*beta)
    if args.rho == np.inf:
        eps_dp_guarantee = np.inf
    else:
        eps_dp_guarantee = compute_epsilon_from_rho_delta(rho_prime, omega, args.delta)


    # print(f"density difference new: {np.abs(densities_original-local_densities).sum():.3f}")
    # print(f"density difference new avg: {(densities_original-local_densities).mean(axis=0)}")

    # print(f"\nOverall (maybe fake) zCDP privacy guarantee: eps={overall_dp_guarantee:4f}, delta={args.delta:.2e}\n")
    print(f"Individual zCDP privacy guarantee: eps={eps_dp_guarantee:4f}, delta={args.delta:.2e}\n")

    # Graph reconstruction-ish attacks.
    print("Running graph reconstruction attacks...")
    # print("With OG densities")
    # top_1_accuracy = some_attack(densities, densities_original)
    # top_k_accuracy = top_k_hits(densities, densities_original, k=10)
    print("with local densities:")

    nonnandensities = np.nan_to_num(vectorized_local_densities, nan=0.0, posinf=0.0, neginf=0.0)
    # top_1_accuracy = some_attack(nonnandensities, densities_original)
    top_1_accuracy = 0
    # top_k_accuracy = top_k_hits(nonnandensities, densities_original, k=10)
    top_k_accuracy = 0

    # Training.
    split_idx = None
    train_on_noiseless_stuff = False
    if train_on_noiseless_stuff:
        if args.dataset == "zinc":
            # acc, roc_auc = train_on_zinc(densities, SVR, C=args.C, kernel=args.kernel, degree=args.degree, gamma=args.gamma)
            # This works well but deadly overfits. I'm doubting you can learn something from the counts.
            # acc, roc_auc = train_on_zinc(densities, RandomForestRegressor, n_estimators=100, min_samples_split=10,
            #                              max_depth=None, random_state=args.seed)
            acc, roc_auc = train_on_zinc(densities, GradientBoostingRegressor, n_estimators=500, learning_rate=0.1, max_depth=10, random_state=args.seed)
            # acc, roc_auc= train_on_zinc(densities, KNeighborsRegressor, n_neighbors=1000, metric='euclidean')
            # acc, roc_auc = train_on_zinc(densities, MLP, input_dim=50, output_dim=1, task_type='regression', lr=0.001,
            #                              hidden_dims=[128, 300, 200, 100, 32], epochs=50, batch_size=128)
        elif args.dataset == "molhiv":
            pass
            # acc, roc_auc = train_on_molhiv(densities, SVR, C=args.C, kernel=args.kernel, degree=args.degree, gamma=args.gamma)
            # acc, roc_auc, split_idx, y_train = train_on_molhiv(densities, RandomForestClassifier, n_estimators=100, min_samples_split=10,
            #                              max_depth=None, random_state=args.seed)
            # acc, roc_auc, split_idx, y_train = train_on_molhiv(densities, KNeighborsClassifier, n_neighbors=1000, weights='uniform', algorithm='auto')
            # acc, roc_auc = train_on_molhiv(densities, MLP, input_dim=50, output_dim=2, epochs=50)
        elif args.dataset == 'csl':
            acc, roc_auc, _, y_train = train_on_csl(densities, KNeighborsClassifier, n_neighbors=10, weights='uniform', algorithm='auto')
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")
        print(f"Noiseless, {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")

    # Here you can use either the local densities or the vectorized local densities.
    # Where the local densities have noisy calibrated to each patter, and the vectorized local densities
    # have noise calibrated to the l2 norm of the local sensitivities for each graph.
    if args.dataset == "molhiv":
        # acc, roc_auc, split_idx, y_train = train_on_molhiv(local_densities, KNeighborsClassifier, n_neighbors=500, weights='uniform', algorithm='auto')
        acc, roc_auc, split_idx, y_train = train_on_molhiv(vectorized_local_densities, KNeighborsClassifier, n_neighbors=500, weights='uniform', algorithm='auto')
        # acc, roc_auc, split_idx, y_train = train_on_molhiv_with_feature_perturbation(vectorized_local_densities, densities_original,
        #                                                             KNeighborsClassifier, n_neighbors=500, weights='uniform', algorithm='auto')
        # acc, roc_auc, split_idx, y_train = train_on_molhiv(local_densities, RandomForestClassifier, n_estimators=100, min_samples_split=10,
        #                              max_depth=None, random_state=args.seed)
        print(f"KNN, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif args.dataset == "molbace":
        acc, roc_auc, split_idx, y_train = train_on_molbace(vectorized_local_densities, KNeighborsClassifier, n_neighbors=10, weights='uniform', algorithm='auto')
        # acc, roc_auc, split_idx, y_train = train_on_molhiv(local_densities, RandomForestClassifier, n_estimators=100, min_samples_split=10,
        #                              max_depth=None, random_state=args.seed)
        print(f"KNN molbace, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif args.dataset == "molpcba":
        acc, roc_auc, split_idx, y_train = train_on_molpcba(vectorized_local_densities, KNeighborsClassifier, n_neighbors=2000, weights='uniform', algorithm='auto')
        # acc, roc_auc, split_idx, y_train = train_on_molhiv(local_densities, RandomForestClassifier, n_estimators=100, min_samples_split=10,
        #                              max_depth=None, random_state=args.seed)
        print(f"KNN molpcba, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif args.dataset == "moltox21":
        acc, roc_auc, split_idx, y_train = train_on_moltox21(vectorized_local_densities, KNeighborsClassifier, n_neighbors=100, algorithm='auto')
        # acc, roc_auc, split_idx, y_train = train_on_moltox21(local_densities, RandomForestClassifier, n_estimators=100, min_samples_split=10,
        #                              max_depth=None, random_state=args.seed)
        acc = np.mean(acc)
        roc_auc = np.mean(roc_auc)
        print(f"KNN molpcba, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
    elif args.dataset == "zinc":
        # acc, roc_auc = train_on_zinc(vectorized_local_densities, SVR, C=args.C, kernel=args.kernel, degree=args.degree, gamma=args.gamma)
        # print(f"SVC, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
        # acc, roc_auc = train_on_zinc(vectorized_local_densities, GradientBoostingRegressor, n_estimators=500, learning_rate=0.1, max_depth=10, random_state=args.seed)
        # acc, roc_auc = train_on_zinc(vectorized_local_densities, RandomForestRegressor, n_estimators=400, min_samples_split=10,
        #                             max_depth=None, random_state=args.seed)
        # feature_zinc(vectorized_local_densities, RandomForestRegressor)
        exit()
    elif args.dataset == "csl":
        acc, roc_auc, _, y_train = train_on_csl(vectorized_local_densities, KNeighborsClassifier, n_neighbors=3, weights='uniform', algorithm='auto')
        print(f"KNN, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
        # Got them through grid search.
        C = 88
        gamma = 10
        acc, roc_auc, _, y_train = train_on_csl(vectorized_local_densities, SVC, C=C, kernel=args.kernel, probability=True, gamma=gamma)
        print(f"SVC, noisy vectorized embeddings {args.run} roc-auc {roc_auc:.4f} acc {acc:.4f}")
        grid_search = False
        if grid_search:
            Cs = np.logspace(start=-5, stop=6, num=20).tolist()
            gammas = np.logspace(start=-5, stop=1, num=7).tolist() + ['scale']
            class_weight = ['balanced']
            param_grid = {'C': Cs, 'gamma': gammas, 'class_weight': class_weight}
            svc = SVC(probability=True)
            _, _, y = load_csl_data("CSL", "data/CSL/")
            print(f"num classes: {len(np.unique(y))}")
            split_idx = int(len(y) * 0.8)
            X_train, X_test, y_train, y_test = train_test_split(
                vectorized_local_densities, y, test_size=0.2, random_state=42, stratify=y
            )
            print("Performing gridsearch")
            grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=0)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            print(f"Best params from grid search: {best_params}")
            # results = pd.DataFrame(grid_search.cv_results_)
            # for _, row in results.iterrows():
            #     print(f"Params: {row['params']}, Mean test score: {row['mean_test_score']:.4f}")

            svc_best = SVC(probability=True, **best_params)
            svc_best.fit(X_train, y_train)
            acc = svc_best.score(X_test, y_test)
            print(f"Test accuracy with best params: {acc:.4f}")


    # This is what a msc in data science gives you.
    print(args.seed)
    output_file = f"{args.output_file}_knn_{args.dataset.upper()}_{args.hom_size}_" \
    f"{args.pattern_count}_s{args.run}_tw{args.max_treewidth}" \
    f".csv" if hasattr(args, 'output_file') and args.output_file else \
    f'results/classifier_on_homcounts/{args.save_path}_{args.dataset.upper()}_{args.hom_size}_' \
    f'{args.pattern_count}_s{args.run}_tw{args.max_treewidth}_rho{args.rho}_' \
    f'delta{args.delta}_seed{args.seed}.csv'

    # Store separability and sensitivity.
    if split_idx is not None:
        sep=0
        # sep = compute_interclass_separability(densities_original[split_idx['train']], y_train, 10)
        # print(f"Separability original: {sep:.4f}")
        # sep = compute_interclass_separability(densities[split_idx['train']], y_train, 10)
        # print(f"Separability noisy: {sep:.4f}")
    else:
        sep = 0


    with open(output_file, 'a') as f:
        f.write(f"{args.run},{args.noise},{args.rho},{args.delta},{roc_auc:.4f},{acc:.4f},{args.seed},{sep:.4f},{np.max(sensitivities)},"
                f"{top_1_accuracy:.5f},{top_k_accuracy},{eps_dp_guarantee},{args.max_treewidth}\n")
        f.flush()


if __name__ == '__main__':
    main()

