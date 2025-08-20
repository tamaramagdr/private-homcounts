import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from ogb.graphproppred import GraphPropPredDataset
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, SVC
from torch.utils.data import TensorDataset
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from feature_perturbation import multi_bit_mechanism
from noise_and_sensitivity import compute_per_element_sensitivies, get_noise
from utils import load_csl_data


class MLP(nn.Module, BaseEstimator):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32], task_type='regression', lr=0.0001, epochs=50,
                 batch_size=32):
        super(MLP, self).__init__()
        self.task_type = task_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
        layers.append(nn.Linear(dims[-1], output_dim))
        if task_type == 'classification':
            layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler(feature_range=(0, 1))

    def forward(self, x):
        return self.model(x)

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
        y_train, y_val = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

        if self.task_type == 'classification':
            y_train, y_val = y_train.long(), y_val.long()

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.SmoothL1Loss()
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(self.epochs/5), gamma=0.1)

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()

            # This had no batches.
            # outputs = self(X_train)
            # # y_train = y_train.view(-1, 1)
            # loss = criterion(outputs, y_train.view(-1, 1))
            # loss.backward()
            # optimizer.step()

            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    print("NaNs in input or target!")
                loss = criterion(outputs, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val)
                val_loss = criterion(val_outputs, y_val.view(-1, 1))
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss.item():.4f}")
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
        if self.task_type == 'classification':
            return torch.argmax(outputs, dim=1).numpy()
        return outputs.numpy()

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        if self.task_type == 'classification':
            y_pred = self.predict(X_test)
            return roc_auc_score(X_test, y_pred)
        return np.abs(y_pred - y_test).mean()

    def accuracy_score(self, X_test, y_test):
        accuracy_score(y_pred=self.predict(X_test), y_true=y_test)


    def predict_proba(self, X):
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only applicable for classification tasks.")
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
        return outputs.numpy()

    def roc_auc(self, X_test, y_test):
        if self.task_type == 'classification':
            # y_pred = self.predict(X_test)
            y_pred = self.predict_proba(X_test)[:, 0]
            return roc_auc_score(y_test, y_pred)
        raise ValueError("ROC AUC is only applicable for classification tasks.")



def train_on_zinc(densities, learning_model, **kwargs):
    train = ZINC(root='data/ZINC', subset=True, split='train')
    val = ZINC(root='data/ZINC', subset=True, split='val')
    test = ZINC(root='data/ZINC', subset=True, split='test')
    y_train = train.data.y
    y_test = test.data.y

    use_features = True
    if use_features:
        # Annoyingly, have to combine things to make the histogram.
        all_graphs = list(train) + list(val) + list(test)
        n_bins = 15
        n_graphs = len(all_graphs)

        # Get overall bin sizes.
        all_vals = np.concatenate([g.x.cpu().numpy().ravel() for g in all_graphs])
        global_min, global_max = all_vals.min(), all_vals.max()
        bin_edges = np.linspace(global_min, global_max, n_bins + 1, dtype=np.float32)

        # Build histogram matrix.
        hist_matrix = np.zeros((n_graphs, n_bins), dtype=np.float32)

        extra_features = []

        for i, g in enumerate(all_graphs):
            node_vals = g.x.cpu().numpy().ravel()
            hist, _ = np.histogram(node_vals, bins=bin_edges)
            hist = hist.astype(np.float32) / hist.sum()
            hist_matrix[i] = hist
            # Also compute some extra features.
            mean_feat = node_vals.mean(axis=0)
            sum_feat = node_vals.sum(axis=0)
            max_feat = node_vals.max(axis=0)
            num_nodes = [g.num_nodes]
            min_feat = node_vals.min()
            std_feat = node_vals.std()
            median_feat = np.median(node_vals)
            skew_feat = scipy.stats.skew(node_vals)
            kurt_feat = scipy.stats.kurtosis(node_vals)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                edge_vals = g.edge_attr.cpu().numpy().ravel()
                mean_edge = edge_vals.mean()
                std_edge = edge_vals.std()
            else:
                mean_edge = std_edge = 0
            feats = np.concatenate([
                np.atleast_1d(mean_feat), np.atleast_1d(std_feat), np.atleast_1d(median_feat),
                np.atleast_1d(skew_feat), np.atleast_1d(kurt_feat),
                # np.atleast_1d(mean_deg), np.atleast_1d(max_deg), np.atleast_1d(min_deg), np.atleast_1d(std_deg),
                # np.atleast_1d(mean_edge), np.atleast_1d(std_edge),
                np.atleast_1d(sum_feat), np.atleast_1d(max_feat),
                np.atleast_1d(num_nodes), np.atleast_1d(min_feat)
            ])
            extra_features.append(feats)

        # Perturb features with multi-bit mechanism.
        # hist_matrix = multi_bit_mechanism(hist_matrix, epsilon=1.0)

        # densities = np.hstack([densities, hist_matrix, extra_features])
        # densities = np.hstack([hist_matrix, extra_features])

    X_train = densities[:10000]
    X_val = densities[10000:11000]
    X_test = densities[11000:]

    print(X_train.shape)
    print(X_train[0])
    # scaler = StandardScaler()
    # scaler = scaler.fit(X_train)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(densities)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    # TODO: make this programmatic.
    acc = np.abs(y_test - clf.predict(X_test)).mean()
    roc_auc = 0

    print(f"train MAE: {np.abs(y_train - clf.predict(X_train)).mean():.4f}")
    print(f"TEST MAE: {np.abs(y_test - clf.predict(X_test)).mean():.4f}")
    print(f"BAD MAE: {np.abs(y_test - clf.predict(X_val)).mean():.4f}")

    return acc, roc_auc


def feature_zinc(densities, learning_model, **kwargs):
    train = ZINC(root='data/ZINC', subset=True, split='train')
    val = ZINC(root='data/ZINC', subset=True, split='val')
    test = ZINC(root='data/ZINC', subset=True, split='test')

    all_graphs = list(train) + list(val) + list(test)
    n_bins = 15
    n_graphs = len(all_graphs)

    # Get overall bin sizes for node features
    all_node_vals = np.concatenate([g.x.cpu().numpy().ravel() for g in all_graphs])
    global_min, global_max = all_node_vals.min(), all_node_vals.max()
    bin_edges = np.linspace(global_min, global_max, n_bins + 1, dtype=np.float32)

    features = []
    for g in all_graphs:
        node_vals = g.x.cpu().numpy().ravel()
        # Node feature histogram
        hist, _ = np.histogram(node_vals, bins=bin_edges)
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        # Node stats
        node_stats = [
            node_vals.mean(), node_vals.std(), np.median(node_vals),
            scipy.stats.skew(node_vals), scipy.stats.kurtosis(node_vals),
            node_vals.sum(), node_vals.max(), node_vals.min(), len(node_vals)
        ]
        # Edge stats (if available)
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            edge_vals = g.edge_attr.cpu().numpy().ravel()
            edge_stats = [
                edge_vals.mean(), edge_vals.std(), np.median(edge_vals)
                # scipy.stats.skew(edge_vals), scipy.stats.kurtosis(edge_vals),
                # edge_vals.sum(), edge_vals.max(), edge_vals.min(), len(edge_vals)
            ]
        else:
            edge_stats = [0.0] * 9
        # Number of nodes and edges
        num_nodes = g.num_nodes
        num_edges = g.num_edges if hasattr(g, 'num_edges') else (g.edge_index.size(1) if hasattr(g, 'edge_index') else 0)
        features.append(np.concatenate([hist, node_stats, edge_stats, [num_nodes, num_edges]]))

    features = np.stack(features)
    # Stack the densities, but only the first 10 dimensions.
    densities = densities[:, :10]
    # features = np.hstack([densities, features])
    # Targets
    y = np.concatenate([g.y.cpu().numpy().ravel() for g in all_graphs])

    # Replace NaNs with 0 in features and targets.
    features = np.nan_to_num(features, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    # Split
    X_train = features[:10000]
    X_val = features[10000:11000]
    X_test = features[11000:]
    y_train = y[:10000]
    y_val = y[10000:11000]
    y_test = y[11000:]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Model
    # reg = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    reg = MLP(input_dim=X_train.shape[1], output_dim=1, hidden_dims=[1024, 516, 256, 128, 64, 32],
              task_type='regression', lr=0.001, epochs=300, batch_size=64)
    reg.fit(X_train, y_train)

    # Evaluate
    print(f"train MAE: {mean_absolute_error(y_train, reg.predict(X_train)):.4f}")
    print(f"val MAE: {mean_absolute_error(y_val, reg.predict(X_val)):.4f}")
    print(f"test MAE: {mean_absolute_error(y_test, reg.predict(X_test)):.4f}")


def train_on_csl(densities, learning_model, **kwargs):
    _, _, y = load_csl_data("CSL", "data/CSL/")
    # TODO: wip

    split_idx = int(len(y)*0.8)

    # X_train = densities[:split_idx]
    # X_test = densities[split_idx:]
    # y_train = y[:split_idx]
    # y_test = y[split_idx:]

    # To have balanced splits.
    X_train, X_test, y_train, y_test = train_test_split(
        densities, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    # TODO: make this programmatic.
    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    # roc_auc = roc_auc_score(y_true=y_test, y_score=clf.decision_function(X_test))
    # This works for random forest and knn
    roc_auc = 9
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test), multi_class='ovr', labels=np.unique(y_train))
    # roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])
    # roc_auc = clf.roc_auc(X_test, y_test)
    #            pedictroc_auc_score(y_true=y_test, y_score=clf.decision_function(X_test)))

    return acc, roc_auc, split_idx, y_train


def train_on_moltox21(densities, learning_model, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-moltox21')
    split_idx = dataset.get_idx_split()

    y = dataset.labels
    X_train = densities[split_idx['train']]
    X_test = densities[split_idx['test']]
    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    accuracies = []
    aucs = []

    # for task in range(y_train.shape[1]):
    for task in range(1):
        y_train_task = y_train[:, task]
        y_test_task = y_test[:, task]

        # Only train if there are positive and negative examples.
        if np.sum(~np.isnan(y_train_task)) > 0 and len(np.unique(y_train_task[~np.isnan(y_train_task)])) > 1:
            clf = learning_model(**kwargs)
            print("If we fits...")
            mask = ~np.isnan(y_train_task)
            clf.fit(X_train[mask][~np.isnan(X_train[mask]).any(axis=1)],
                    y_train_task[mask][~np.isnan(X_train[mask]).any(axis=1)])
            # clf.fit(X_train[~np.isnan(y_train_task)], y_train_task[~np.isnan(y_train_task)])
            print("...we sits!")

            # Print some statistics.
            num_not_nan = np.sum(~np.isnan(y[:, task]))
            print(f"Number of non-NaN training samples: {num_not_nan}")

            num_nodes = []
            num_edges = []

            for i in range(len(dataset)):
                graph, _ = dataset[i]
                num_nodes.append(graph['num_nodes'])
                num_edges.append(graph['edge_index'].shape[1])

            avg_nodes = np.mean(num_nodes)
            avg_edges = np.mean(num_edges)
            print(f"Average number of nodes per graph (all): {avg_nodes:.2f}")
            print(f"Average number of edges per graph (all): {avg_edges:.2f}")

            # mask out NaN targets in validation set
            mask = ~np.isnan(y_test_task)
            acc = accuracy_score(y_true=y_test_task[mask], y_pred=clf.predict(X_test)[mask])
            roc_auc = roc_auc_score(y_true=y_test_task[mask], y_score=clf.predict_proba(X_test)[:, 1][mask])
            print(f"Task {task}, accuracy: {acc:.4f}, auc: {roc_auc:.4f}")
            accuracies.append(acc)
            aucs.append(roc_auc)
        else:
            accuracies.append(np.nan)
            aucs.append(np.nan)

    return accuracies, aucs, split_idx, y_train


def train_on_molpcba(densities, learning_model, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molpcba')
    split_idx = dataset.get_idx_split()

    y = dataset.labels
    X_train = densities[split_idx['train']]
    X_test = densities[split_idx['test']]
    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Consider the first task only.
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    clf = learning_model(**kwargs)

    print("If we fits...")
    mask = ~np.isnan(y_train)
    clf.fit(X_train[mask][~np.isnan(X_train[mask]).any(axis=1)],
            y_train[mask][~np.isnan(X_train[mask]).any(axis=1)])
    print("...we sits!")

    mask = ~np.isnan(y_test)
    acc = accuracy_score(y_pred=clf.predict(X_test)[mask], y_true=y_test[mask])
    print("ACCURACY:", acc)
    roc_auc = roc_auc_score(y_true=y_test[mask], y_score=clf.predict_proba(X_test)[:, 1][mask])

    return acc, roc_auc, split_idx, y_train


def train_on_molbace(densities, learning_model, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molbace')
    split_idx = dataset.get_idx_split()

    y = dataset.labels.ravel()
    X_train = densities[split_idx['train']]
    X_test = densities[split_idx['test']]
    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]


    # Print some statistics.
    num_not_nan = np.sum(~np.isnan(y))
    print(f"Number of non-NaN training samples: {num_not_nan}")

    num_nodes = []
    num_edges = []

    for i in range(len(dataset)):
        graph, _ = dataset[i]
        num_nodes.append(graph['num_nodes'])
        num_edges.append(graph['edge_index'].shape[1])

    avg_nodes = np.mean(num_nodes)
    avg_edges = np.mean(num_edges)
    print(f"Average number of nodes per graph (all): {avg_nodes:.2f}")
    print(f"Average number of edges per graph (all): {avg_edges:.2f}")

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    # TODO: make this programmatic.
    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    print("ACCURACY:", acc)
    # roc_auc = roc_auc_score(y_true=y_test, y_score=clf.decision_function(X_test))
    # This works for random forest and knn
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])
    # roc_auc = clf.roc_auc(X_test, y_test)
    #            pedictroc_auc_score(y_true=y_test, y_score=clf.decision_function(X_test)))

    return acc, roc_auc, split_idx, y_train


def train_on_molhiv(densities, densities_original, learning_model, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molhiv')
    split_idx = dataset.get_idx_split()

    y = dataset.labels.ravel()

    # this block is all copy-pasted from https://github.com/pwelke/homcount
    Cs = np.logspace(start=-5, stop=6, num=20).tolist()
    gammas = np.logspace(start=-5, stop=1, num=7).tolist() + ['scale']
    class_weight = ['balanced']
    param_grid = {'C': Cs, 'gamma': gammas, 'class_weight': class_weight}

    X_train = densities[split_idx['train']]
    # X_test = densities[split_idx['test']]
    X_test = densities_original[split_idx['test']]
    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search = False
    if grid_search:
        print("Performing grid search")
        # grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=gs_nfolds, n_jobs=8)
        # grid_search.fit(X_train, y_train)
        # if args.verbose:
        #     print(grid_search.best_params_)
        # with open(f'results/classifier_on_homcounts/grid_search_best_params_{args.run}.txt', 'w') as f:
        #     f.write(str(grid_search.best_params_))
        # clf = SVC(**grid_search.best_params_)
    else:
        print("No grid search")
        # clf = SVC(**kwargs, decision_function_shape='ovr',
        #           random_state=None, class_weight='balanced')

    # clf = KNeighborsClassifier(n_neighbors=1000, weights='uniform', algorithm='auto')
    # clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, class_weight='balanced')
    # clf = GaussianNB()
    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    # TODO: make this programmatic.
    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    print("ACCURACY:", acc)
    # roc_auc = roc_auc_score(y_true=y_test, y_score=clf.decision_function(X_test))
    # This works for random forest and knn
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])
    # roc_auc = clf.roc_auc(X_test, y_test)
    #            pedictroc_auc_score(y_true=y_test, y_score=clf.decision_function(X_test)))

    return acc, roc_auc, split_idx, y_train

def train_on_molhiv_with_feature_perturbation(densities, learning_model, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molhiv')
    split_idx = dataset.get_idx_split()


    # Get the features for each graph, and put them in a histogram with n_bins bins.
    n_bins = 15
    n_graphs = len(dataset)

    # Get overall bin sizes.
    all_vals = np.concatenate([dataset[i][0]["node_feat"].ravel() for i in range(n_graphs)])
    global_min, global_max = all_vals.min(), all_vals.max()
    bin_edges = np.linspace(global_min, global_max, n_bins + 1, dtype=np.float32)

    # Build histogram matrix.
    hist_matrix = np.zeros((n_graphs, n_bins), dtype=np.float32)

    for i in range(n_graphs):
        node_vals = dataset[i][0]["node_feat"].ravel()
        # counts, one can use density=True for probabilities.
        hist, _ = np.histogram(node_vals, bins=bin_edges)
        # Normalise so every row sums to 1.
        hist = hist.astype(np.float32) / hist.sum()
        hist_matrix[i] = hist


    # Perturb features with multi-bit mechanism.
    hist_matrix = multi_bit_mechanism(hist_matrix, epsilon=1.0)

    densities = np.hstack([densities, hist_matrix])

    y = dataset.labels.ravel()

    # this block is all copy-pasted from https://github.com/pwelke/homcount
    Cs = np.logspace(start=-5, stop=6, num=20).tolist()
    gammas = np.logspace(start=-5, stop=1, num=7).tolist() + ['scale']
    class_weight = ['balanced']
    param_grid = {'C': Cs, 'gamma': gammas, 'class_weight': class_weight}

    X_train = densities[split_idx['train']]
    X_test = densities[split_idx['test']]
    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search = False
    if grid_search:
        print("Performing grid search")
        # grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=gs_nfolds, n_jobs=8)
        # grid_search.fit(X_train, y_train)
        # if args.verbose:
        #     print(grid_search.best_params_)
            # with open(f'results/classifier_on_homcounts/grid_search_best_params_{args.run}.txt', 'w') as f:
            #     f.write(str(grid_search.best_params_))
        # clf = SVC(**grid_search.best_params_)
    else:
        print("No grid search")
        # clf = SVC(**kwargs, decision_function_shape='ovr',
        #           random_state=None, class_weight='balanced')

    # clf = KNeighborsClassifier(n_neighbors=1000, weights='uniform', algorithm='auto')
    # clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, class_weight='balanced')
    # clf = GaussianNB()
    clf = learning_model(**kwargs)


    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    # TODO: make this programmatic.
    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    print("ACCURACY:", acc)
    # roc_auc = roc_auc_score(y_true=y_test, y_score=clf.decision_function(X_test))
    # This works for random forest and knn
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])
    # roc_auc = clf.roc_auc(X_test, y_test)
    #            pedictroc_auc_score(y_true=y_test, y_score=clf.decision_function(X_test)))

    return acc, roc_auc, split_idx, y_train


def do_many_things_and_plot(y_trai, y_test, densities_original, counts, pattern_edges, graph_sizes, dataset_length, split_idx, delta, save_path):
    with open(f'results/classifier_on_homcounts/{save_path}_grid.csv', 'w') as f:
        pass
    for epsilon in range(1, 100, 3):
        # for epsilon in [0.01, 0.1, 0.5] + list(range(1, 100, 3)):
        for seed in range(1, 2):
            torch.manual_seed(seed)
            sensitivities, variances = compute_per_element_sensitivies(pattern_edges, graph_sizes, epsilon, delta)
            stds = [np.sqrt(variances[i]) for i in range(len(variances))]
            noise = get_noise(counts, dataset_length, stds)
            densities = densities_original + noise
            X_train = densities[split_idx['train']]
            X_test = densities[split_idx['test']]

            # B_matrix = pairwise_bhattacharyya_diagcov(X_train, stds)
            # avg_overlap = average_overlap(B_matrix)
            # print(f"epsilon: {epsilon}, avg overlap: {avg_overlap}")

            clf = KNeighborsClassifier(n_neighbors=1000, weights='uniform', algorithm='auto')
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
            roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])
            with open(f'results/classifier_on_homcounts/{args.save_path}_grid.csv', 'a') as f:
                f.write(f"{epsilon},{roc_auc:.4f},{acc:.4f}{seed}\n")
                f.flush()

    grid_results = pd.read_csv(f'results/classifier_on_homcounts/{save_path}_grid.csv', header=None, names=["epsilon", "roc_auc", "acc"])
    plt.figure(figsize=(10, 6))
    plt.scatter(grid_results["roc_auc"], grid_results["epsilon"], label="ROC AUC", marker="o")
    # plt.plot(grid_results["epsilon"], grid_results["acc"], label="Accuracy", marker="x")
    # plt.xscale("log")  # Log scale for epsilon
    plt.xlabel("Roc auc")
    plt.ylabel("epsilon")
    plt.title("Epsilon vs ROC AUC")
    plt.legend()
    # Some exponential fitting.
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c
    popt, _ = curve_fit(exponential_func, grid_results["roc_auc"], grid_results["epsilon"], maxfev=10000)

    # Plot the fitted curve
    x_fit = np.linspace(min(grid_results["roc_auc"]), max(grid_results["roc_auc"]), 500)
    y_fit = exponential_func(x_fit, *popt)
    plt.plot(x_fit, y_fit, label="Exponential Fit", color="red")

    plt.grid(True)
    plt.show()

