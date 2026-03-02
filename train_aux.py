import numpy as np
from ogb.graphproppred import GraphPropPredDataset
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import TUDataset


def train_on_molbace_with_features(densities, densities_original, learning_model, use_features=True, features_only=False, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molbace')
    split_idx = dataset.get_idx_split()

    # Node features stuff.
    all_graphs = [dataset[i][0]['node_feat'].ravel() for i in range(len(dataset))]
    features = np.stack([
        np.array([
            g.mean(),
            g.std(),
            np.median(g),
            g.min(),
            g.max(),
            g.sum(),
            g.shape[0]
        ])
        for g in all_graphs
    ])

    # Targets and replace NaNs with 0.
    features = np.nan_to_num(features, nan=0.0)

    # Split
    train_features = features[split_idx['train']]
    test_features = features[split_idx['test']]
    y = dataset.labels.ravel()

    if features_only:
        pca = PCA(n_components=7)
        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
        X_train = train_features
        X_test = test_features
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        print("Not with features only...")
        train_densities = densities[split_idx['train']]
        test_densities = densities_original[split_idx['test']]
        X_train = np.hstack([train_densities, train_features])
        X_test = np.hstack([test_densities, test_features])
        pca = PCA(n_components=9)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    return acc, roc_auc, split_idx, y_train

def train_on_molbace(densities, densities_original, learning_model, use_features=False, features_only=False, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molbace')
    split_idx = dataset.get_idx_split()

    y = dataset.labels.ravel()
    X_train = densities[split_idx['train']]
    X_test = densities_original[split_idx['test']]
    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]


    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    return acc, roc_auc, split_idx, y_train

def train_on_sbm(densities, labels, learning_model, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(
        densities, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    # TODO: make this programmatic.
    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    print("ACCURACY:", acc)
    roc_auc = 0
    if len(np.unique(labels)) == 2:
        roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    return acc, roc_auc


def train_on_molbbbp(densities, densities_original, learning_model, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molbbbp')
    split_idx = dataset.get_idx_split()

    y = dataset.labels.ravel()

    X_train = densities[split_idx['train']]
    X_test = densities_original[split_idx['test']]
    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

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
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    return acc, roc_auc, split_idx, y_train



def train_on_molbbbp_with_features(densities, densities_original, learning_model,
                                  fetures_only=False, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molbbbp')
    split_idx = dataset.get_idx_split()

    all_graphs = [dataset[i][0]['node_feat'].ravel() for i in range(len(dataset))]

    features = np.stack([
        np.array([
            g.mean(),
            g.std(),
            np.median(g),
            g.min(),
            g.max(),
            g.sum(),
            g.shape[0]  # Number of nodes?
        ])
        for g in all_graphs
    ])

    # Targets and replace NaNs with 0.
    features = np.nan_to_num(features, nan=0.0)

    # Split
    train_features = features[split_idx['train']]
    test_features = features[split_idx['test']]

    y = dataset.labels.ravel()

    if fetures_only:
        pca = PCA(n_components=7)
        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
        X_train = train_features
        X_test = test_features
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        print("Not with features only...")
        train_densities = densities[split_idx['train']]
        test_densities = densities_original[split_idx['test']]
        X_train = np.hstack([train_densities, train_features])
        X_test = np.hstack([test_densities, test_features])
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    return acc, roc_auc, split_idx, y_train

def train_on_molhiv(densities, densities_original, learning_model, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molhiv')
    split_idx = dataset.get_idx_split()

    y = dataset.labels.ravel()

    X_train = densities[split_idx['train']]
    X_test = densities_original[split_idx['test']]
    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    return acc, roc_auc, split_idx, y_train

def train_on_molhiv_with_features(densities, densities_original, learning_model,
                                  fetures_only=False, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-molhiv')
    split_idx = dataset.get_idx_split()

    all_graphs = [dataset[i][0]['node_feat'].ravel() for i in range(len(dataset))]

    features = np.stack([
        np.array([
            g.mean(),
            g.std(),
            np.median(g),
            g.min(),
            g.max(),
            g.sum(),
            g.shape[0]  # Number of nodes?
        ])
        for g in all_graphs
    ])

    # Targets and replace NaNs with 0.
    features = np.nan_to_num(features, nan=0.0)

    # Split
    train_features = features[split_idx['train']]
    test_features = features[split_idx['test']]

    y = dataset.labels.ravel()

    if fetures_only:
        pca = PCA(n_components=7)
        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
        X_train = train_features
        X_test = test_features
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        print("Not with features only...")
        train_densities = densities[split_idx['train']]
        test_densities = densities_original[split_idx['test']]
        X_train = np.hstack([train_densities, train_features])
        X_test = np.hstack([test_densities, test_features])
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    return acc, roc_auc, split_idx, y_train

def train_on_mollipo(densities, densities_original, learning_model, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-mollipo')
    split_idx = dataset.get_idx_split()

    y = dataset.labels.ravel()

    X_train = densities[split_idx['train']]
    X_test = densities_original[split_idx['test']]
    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    y_pred = clf.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print("RMSE:", rmse)
    roc_auc = 0  # Not applicable for regression

    return rmse, roc_auc, split_idx, y_train

def train_on_mollipo_with_features(densities, densities_original, learning_model,
                                  fetures_only=False, **kwargs):
    dataset = GraphPropPredDataset(name='ogbg-mollipo')
    split_idx = dataset.get_idx_split()

    all_graphs = [dataset[i][0]['node_feat'].ravel() for i in range(len(dataset))]

    features = np.stack([
        np.array([
            g.mean(),
            g.std(),
            np.median(g),
            g.min(),
            g.max(),
            g.sum(),
            g.shape[0]  # Number of nodes?
        ])
        for g in all_graphs
    ])

    # Targets and replace NaNs with 0.
    features = np.nan_to_num(features, nan=0.0)

    # Split
    train_features = features[split_idx['train']]
    test_features = features[split_idx['test']]

    y = dataset.labels.ravel()

    if fetures_only:
        pca = PCA(n_components=7)
        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
        X_train = train_features
        X_test = test_features
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        print("Not with features only...")
        train_densities = densities[split_idx['train']]
        test_densities = densities_original[split_idx['test']]
        X_train = np.hstack([train_densities, train_features])
        X_test = np.hstack([test_densities, test_features])
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


    y_train = y[split_idx['train']]
    y_test = y[split_idx['test']]

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    y_pred = clf.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print("RMSE:", rmse)
    roc_auc = 0  # Not applicable for regression

    return rmse, roc_auc, split_idx, y_train





def train_on_reddit_binary(densities, densities_original, learning_model, **kwargs):

    dataset = TUDataset(root="data/TUDataset", name="REDDIT-BINARY")
    y = dataset.data.y.numpy().ravel()


    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42)

    X_train = densities[train_idx]
    X_test = densities_original[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    pca = PCA(n_components=5)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    return acc, roc_auc, 0, y_train


def train_on_reddit_multi(densities, densities_original, learning_model, **kwargs):

    dataset = TUDataset(root="data/TUDataset", name="REDDIT-MULTI-5K")
    y = dataset.data.y.numpy().ravel()


    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

    X_train = densities[train_idx]
    X_test = densities_original[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    pca = PCA(n_components=5)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    # roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)
        if y_score.ndim == 1 or (hasattr(y_score, "shape") and y_score.shape[1] == 2):
            roc_auc = roc_auc_score(y_true=y_test, y_score=y_score[:, 1])
        else:
            roc_auc = roc_auc_score(y_true=y_test, y_score=y_score, multi_class="ovr", average="macro")
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
        if y_score.ndim == 1 or (hasattr(y_score, "shape") and y_score.shape[1] == 2):
            roc_auc = roc_auc_score(y_true=y_test, y_score=y_score[:, 1])
        else:
            roc_auc = roc_auc_score(y_true=y_test, y_score=y_score, multi_class="ovr", average="macro")
    else:
        roc_auc = 0

    return acc, roc_auc, 0, y_train


def train_on_github(densities, densities_original, learning_model, **kwargs):

    dataset = TUDataset(root="data/TUDataset", name="github_stargazers")
    y = dataset.data.y.numpy().ravel()

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42)

    X_train = densities[train_idx]
    X_test = densities_original[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Do a pca of X_train and X_test to reduce dimensionality.
    # pca = PCA(n_components=10)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # Fit a scaler to training data.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = learning_model(**kwargs)

    print("If we fits...")
    clf.fit(X_train, y_train)
    print("...we sits!")

    acc = accuracy_score(y_pred=clf.predict(X_test), y_true=y_test)
    roc_auc = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1])

    return acc, roc_auc, 0, y_train
