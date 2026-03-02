import os
import argparse
import pickle
import shutil

import networkx as nx
import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader, Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_mean_pool
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.transforms import BaseTransform

import numpy as np
from torch_geometric.utils import to_networkx, from_networkx, to_undirected


class SimpleEvaluator:
    def __init__(self, task_type):
        # Mimic OGB’s interface
        self.task_type = task_type
        if task_type == "binary classification":
            self.eval_metric = "rocauc"
        elif task_type == "multiclass classification":
            self.eval_metric = "acc"
        else:
            self.eval_metric = "rmse"

    def eval(self, input_dict):
        y_true = input_dict["y_true"].view(-1).numpy()
        y_pred = input_dict["y_pred"].view(-1).numpy()
        if self.task_type == "binary classification":
            y_prob = 1 / (1 + np.exp(-y_pred))
            return {"rocauc": roc_auc_score(y_true, y_prob)}
        elif self.task_type == "multiclass classification":
            y_pred_labels = np.argmax(y_pred, axis=1)
            return {"acc": (y_true == y_pred_labels).sum() / len(y_true)}
        else:
            return {"rmse": mean_squared_error(y_true, y_pred, squared=False)}

class CustomDataset(Dataset):
    def __init__(self, tu_dataset_name=None, graphs_path=None, labels_path=None, pre_transform=None,
                 seed=0, task_type="binary classification"):
        if tu_dataset_name is not None:
            dataset = TUDataset(root="data/TUDataset", name=tu_dataset_name)
            self.y = dataset.data.y.numpy().ravel()
            self.graphs = [to_networkx(g, to_undirected=True) for g in dataset]
        else:
            with open(labels_path, "rb") as f:
                self.y = pickle.load(f)
            with open(graphs_path, "rb") as f:
                self.graphs = pickle.load(f)

        # Make tensor.
        self.y = torch.as_tensor(self.y)

        if isinstance(self.graphs[0], nx.Graph):
            new_graphs = []
            for g in self.graphs:
                data = from_networkx(g)
                # No node features: use all-ones.
                data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
                new_graphs.append(data)
            self.graphs = new_graphs

        # Assume y is shape [N] or [N, 1].
        if self.y.ndim == 1:
            self.num_tasks = 1
        else:
            self.num_tasks = self.y.shape[1]
        self.num_node_features = self.graphs[0].x.size(1)
        self.task_type = task_type

        self.pre_transform = pre_transform
        self._make_split(seed)

    def __len__(self):
        return len(self.graphs)

    # Fixed 70 / 10 / 20 split.
    def _make_split(self, seed):
        n = len(self.y)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)

        n_train = int(0.7 * n)
        n_valid = int(0.1 * n)

        idx_train = perm[:n_train]
        idx_valid = perm[n_train:n_train + n_valid]
        idx_test  = perm[n_train + n_valid:]

        self._split_idx = {
            "train": idx_train,
            "valid": idx_valid,
            "test": idx_test,
        }

    def get_idx_split(self):
        return self._split_idx

    def __getitem__(self, idx):
        # Multiple indices (tensor/list/ndarray).
        if isinstance(idx, (list, tuple, torch.Tensor, np.ndarray)):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            elif isinstance(idx, np.ndarray):
                idx = idx.tolist()
            return [self[i] for i in idx]

        # Single index.
        i = int(idx)
        data = self.graphs[i]
        data.y = self.y[i].view(-1)  # Ensure 1D tensor.

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data



@functional_transform('deg_pres_randomized_response')
class DegPresRandomizedResponse(BaseTransform):
    r"""Degree-Preserving Randomized Response (DPRR) as in
    "Degree-Preserving Randomized Response for Graph Neural Networks under Local Differential Privacy"
    (Hidano & Murakami, 2024), Algorithm 1.

    Args:
        eps1 (float): Privacy budget for the Laplace mechanism on degrees.
        eps2 (float): Privacy budget for Warner's Randomized Response.
                      Total edge-LDP epsilon is eps1 + eps2.
    """
    def __init__(self, eps1: float, eps2: float):
        if eps1 <= 0 or eps2 <= 0:
            raise ValueError("eps1 and eps2 must be positive.")
        self.eps1 = float(eps1)
        self.eps2 = float(eps2)

        # RR's p = e^{eps2} / (e^{eps2} + 1).
        self.p = np.exp(self.eps2) / (np.exp(self.eps2) + 1)

    def __call__(self, data: Data) -> Data:
        g = to_networkx(data).to_undirected()
        nodes = list(g.nodes())
        n = len(nodes)

        # Prepare a DIRECTED noisy adjacency (n x n boolean),
        # where row i is the noisy neighbor list tilde a_i.
        noisy_dir = np.zeros((n, n), dtype=bool)

        # Laplace noise scale.
        lap_scale = 1.0 / self.eps1
        p = self.p

        for i, u in enumerate(nodes):
            # 1) True degree d_i.
            d_i = g.degree[u]

            # 2) Noisy degree d_i*.
            d_star = d_i + np.random.laplace(loc=0.0, scale=lap_scale)

            # 3) Compute sampling probability q_i.
            denom = d_star * (2 * p - 1.0) + (n - 1) * (1.0 - p)
            if denom <= 0:
                # Numerically nasty corner case: fall back to q_i = 0 (i.e., no edges kept).
                q_i = 0.0
            else:
                q_i = d_star / denom

            # Project q_i to [0, 1].
            q_i = float(max(0.0, min(1.0, q_i)))

            # 4) Apply RR + edge sampling to row i (node v_i).
            for j, v in enumerate(nodes):
                if i == j:
                    continue  # No self-loops.

                # Original bit a_{i,j}.
                a_ij = 1 if g.has_edge(u, v) else 0

                # RR: send bit as-is with prob p, otherwise flip.
                if np.random.rand() < p:
                    bit_rr = a_ij
                else:
                    bit_rr = 1 - a_ij

                # Edge sampling: only for 1s after RR.
                if bit_rr == 1 and np.random.rand() < q_i:
                    noisy_dir[i, j] = True

        # Symmetrize to get undirected edges:
        # we add edge {i,j} if either i->j or j->i is present.
        h = nx.Graph()
        h.add_nodes_from(nodes)

        for i in range(n):
            for j in range(i + 1, n):
                if noisy_dir[i, j] or noisy_dir[j, i]:
                    u, v = nodes[i], nodes[j]
                    h.add_edge(u, v)

        G = from_networkx(h)

        G.edge_index = to_undirected(G.edge_index, num_nodes=G.num_nodes)

        # Features / labels, whatever.
        G.x = data.x
        G.y = data.y

        return G


@functional_transform('randomized_response')
class RandomizedResponse(BaseTransform):
    def __init__(self, privacy_budget):
        self.p_make_edge = 1 / (np.exp(privacy_budget) + 1)
        self.p_keep_edge = 1 - self.p_make_edge

    def __call__(self, data: Data) -> Data:
        g = to_networkx(data).to_undirected()
        h = nx.Graph()
        h.add_nodes_from(list(g.nodes()))

        # randomized response
        for i in range(g.number_of_nodes()):
            for j in range(i + 1, g.number_of_nodes()):
                u, v = list(g.nodes())[i], list(g.nodes())[j]
                if g.has_edge(u, v):
                    # keep existing edge
                    if np.random.random() < self.p_keep_edge:
                        h.add_edge(u, v)
                else:
                    # make new edge 
                    if np.random.random() < self.p_make_edge:
                        h.add_edge(u, v)

        G = from_networkx(h)
        G.edge_index = to_undirected(G.edge_index, num_nodes=G.num_nodes)
        G.x = data.x
        G.y = data.y
        return G



class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
        super().__init__()
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_channels, out_channels))
        else:
            layers.append(nn.Linear(in_channels, hidden_channels))
            layers.append(nn.ReLU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels=300, num_layers=5, out_dim=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            mlp = MLP(in_channels if i == 0 else hidden_channels,
                      hidden_channels,
                      hidden_channels,
                      n_layers=2)
            conv = GINConv(mlp)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_dim)
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.head(x)


def train(model, loader, optimizer, device, task_type):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)

        if task_type == "binary classification":
            target = data.y.float().view(out.shape)
            loss = F.binary_cross_entropy_with_logits(out, target, reduction="mean")
        elif task_type == "multiclass classification":
            target = data.y.view(-1)
            loss = F.cross_entropy(out, target, reduction="mean")
        else:  # Regression.
            target = data.y.float().view(out.shape)
            loss = F.l1_loss(out, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, evaluator, task_type):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)

        if "classification" in task_type:
            y_true.append(data.y.view(out.shape))
            y_pred.append(out)
        else:  # Regression.
            y_true.append(data.y.view(out.shape))
            y_pred.append(out)

    y_true = torch.cat(y_true, dim=0).cpu()
    y_pred = torch.cat(y_pred, dim=0).cpu()

    if task_type == "binary classification":
        return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]
    elif task_type == "multiclass classification":
        return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["acc"]
    else:  # regression
        return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rmse"]


def main(dataset_name="ogbg-molhiv", tu_dataset=False):

    parser = argparse.ArgumentParser()
    parser.add_argument("--rr", action="store_true", default=False)
    parser.add_argument("--deg_preserving", action="store_true", default=False)
    parser.add_argument("--eps", type=float, default=1)

    args = parser.parse_args()

    eps = args.eps
    name_rr = "_rr" if args.rr else ""
    name_deg = "_deg_preserving" if args.deg_preserving else ""
    print(f"[INFO] Using eps={eps}")
    rr_tmp = RandomizedResponse(privacy_budget=eps)
    print(f"[INFO] RR p_keep={rr_tmp.p_keep_edge:.6f}, p_make={rr_tmp.p_make_edge:.6f}")

    with open(f"r_{dataset_name}{name_rr}{name_deg}", 'w') as f:
        f.writelines(f"seed,eps,utility\n")

    best_test_list = []

    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Delete pre_transformed graphs for each seed.
        proc = f"data_rr/"
        if os.path.exists(proc):
            shutil.rmtree(proc)

        proc = f"data_clean/"
        if os.path.exists(proc):
            shutil.rmtree(proc)

        proc = f"data_rr_deg_priv/"
        if os.path.exists(proc):
            shutil.rmtree(proc)

        if tu_dataset:
            dataset = CustomDataset(
                tu_dataset_name=dataset_name.upper(),
                pre_transform=None)
            if args.deg_preserving:
                print(f"Running Degree-Preserving Randomized Response on {dataset_name}")
                # Compute the max number of nodes in the dataset.
                max_nodes = 0
                for data in dataset:
                    num_nodes = data.num_nodes
                    if num_nodes > max_nodes:
                        max_nodes = num_nodes
                eps1 = np.max([0.1, np.sqrt(8/(max_nodes-1))] )
                print("\n\nEpsilon 1 for degree perturbation:", eps1)
                eps2 = 1-eps1
                print("\n\nEpsilon 2 for degree perturbation:", eps2)
                dataset_rr = CustomDataset(
                    tu_dataset_name=dataset_name.upper(),
                    pre_transform=DegPresRandomizedResponse(eps1=eps1, eps2=eps2)
                )
            else:
                print(f"Standard Randomized Response on {dataset_name}")
                dataset_rr = CustomDataset(
                    tu_dataset_name=dataset_name.upper(),
                    pre_transform=RandomizedResponse(privacy_budget=eps)
                )

        if not tu_dataset:
            dataset = PygGraphPropPredDataset(name=dataset_name, root="data_clean")
            # remove node features to allow for fair comparison
            dataset.data.x = torch.ones(size=(dataset.data.x.shape[0], 1))
            if args.rr:
                if args.deg_preserving:
                    print("Using Degree-Preserving Randomized Response")
                    # Compute the max number of nodes in the dataset.
                    max_nodes = 0
                    for data in dataset:
                        num_nodes = data.num_nodes
                        if num_nodes > max_nodes:
                            max_nodes = num_nodes
                    eps1 = np.max([0.1, np.sqrt(8/(max_nodes-1))] )
                    print("\n\nEpsilon 1 for degree perturbation:", eps1)
                    eps2 = 1-eps1
                    print("\n\nEpsilon 2 for degree perturbation:", eps2)
                    dataset_rr = PygGraphPropPredDataset(name=dataset_name,
                                                         # Here eps=eps1+eps2, with most allocated for RR.
                                                         # Follows the paper's heuristic for a molecular-sized dataset.
                                                         pre_transform=DegPresRandomizedResponse(eps1=eps1, eps2=eps2),
                                                         root="data_rr_deg_priv")
                else:
                    print("Using Randomized Response")
                    dataset_rr = PygGraphPropPredDataset(name=dataset_name,
                                                         pre_transform=RandomizedResponse(privacy_budget=eps),
                                                         root="data_rr")

                dataset_rr.data.x = torch.ones(size=(dataset_rr.data.x.shape[0], 1))

        split_idx = dataset.get_idx_split()

        if args.rr:
            print("Using perturbed graphs for training.")
            train_loader = DataLoader(dataset_rr[split_idx["train"]], batch_size=64, shuffle=True)
            valid_loader = DataLoader(dataset_rr[split_idx["valid"]], batch_size=64)
        else:
            print("Using original graphs for training.")
            train_loader = DataLoader(dataset[split_idx["train"]], batch_size=64, shuffle=True)
            valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=64)

        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=64)

        in_channels = dataset.num_node_features
        out_dim = dataset.num_tasks
        task_type = dataset.task_type

        model = GIN(in_channels=in_channels, hidden_channels=300, num_layers=5, out_dim=out_dim).to(device)
        if args.deg_preserving:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


        if tu_dataset:  # Custom dataset, so use custom evaluation.
            evaluator = SimpleEvaluator(task_type=task_type)
        else:
            evaluator = Evaluator(name=dataset_name)

        if task_type == "binary classification":
            best_val, best_test = -1, -1
        elif task_type == "multiclass classification":
            best_val, best_test = -1, -1
        else:
            best_val, best_test = float("inf"), float("inf")
        
        for epoch in range(1, 101):
            train_loss = train(model, train_loader, optimizer, device, task_type)
            val_metric = evaluate(model, valid_loader, device, evaluator, task_type)
            test_metric = evaluate(model, test_loader, device, evaluator, task_type)

            improved = (val_metric > best_val) if task_type == "binary classification"  or task_type == "multiclass classification" else (val_metric < best_val)
            if improved:
                best_val, best_test = val_metric, test_metric

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                    f"Val {evaluator.eval_metric}: {val_metric:.4f} | "
                    f"Test {evaluator.eval_metric}: {test_metric:.4f}")

        print(f"Best Val {evaluator.eval_metric}: {best_val:.4f} | "
            f"Corresponding Test {evaluator.eval_metric}: {best_test:.4f}")

        best_test_list.append(best_test)

        with open(f"r_{dataset_name}{name_rr}{name_deg}", 'a') as f:
            f.writelines(f"{seed},{eps},{best_test:.4f}\n")

    with open(f"r_{dataset_name}{name_rr}{name_deg}", 'a') as f:
        f.writelines(f"total,{eps},{np.mean(np.array(best_test_list))} with std {np.std(np.array(best_test_list))}")

if __name__ == "__main__":
    for name in ["ogbg-molbace", "ogbg-molbbbp", "ogbg-mollipo", "ogbg-molhiv"]:
        print(f"\n==== Running {name} ====")
        main(name)
    for name in ["reddit-binary"]:
        print(f"\n==== Running {name} ====")
        main(name, tu_dataset=True)
