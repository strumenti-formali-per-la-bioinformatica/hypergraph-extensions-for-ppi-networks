import argparse
import torch
import torch_geometric
import torch_geometric.datasets
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from time import time
import numpy as np
import networkx as nx
from itertools import combinations
import torch_geometric.transforms as T
from itertools import chain
# Models used for link prediction
from models.lp.gcn import Model as LP_GCN
from models.lp.hypergcn import Model as LP_HyperGCN

import os.path as osp

def main(model_name: str, random_features: bool, score_function):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.set_default_device(device)
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PPI')
    data_train = torch_geometric.datasets.PPI(root=path, split='train')
    data_val = torch_geometric.datasets.PPI(root=path, split='val')
    data_test = torch_geometric.datasets.PPI(root=path, split='test')

    transform = T.RandomLinkSplit(is_undirected=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    for i, data in enumerate(chain(data_train, data_val, data_test)):
        print(f'Graph {i}')
        for experiment in range(5):
            train_data, val_data, test_data = transform(data)
            G = nx.from_edgelist(train_data.edge_index.t().tolist())

            if model_name == 'hypergcn':
                cliques = list(nx.find_cliques(G))
                hyperedges = cliques

                jcs = []
                total_jc = 0
                for clique in cliques:
                    jc = sum(map(lambda x: x[2], score_function(G, list(combinations(clique, 2))))) / len(clique)
                    jcs.append(jc)
                    total_jc += jc
                avg_jc = total_jc / len(list(cliques))
                hyperedges = [clique for jc, clique in zip(jcs, cliques) if jc > avg_jc]

                edge_index = torch.tensor([
                    [n for e in hyperedges for n in e ],
                    [i for i, e in enumerate(hyperedges) for n in e]
                ])
            else:
                edge_index = None
            results = []
            times = []
            history = {
                "train": {
                    "loss": [],
                    "roc_auc": []
                },
                "val": {
                    "loss": [],
                    "roc_auc": []
                },
            }

            X_train = train_data.x
            X_val = val_data.x
            X_test = test_data.x

            if random_features:
                print('Random features')
                X_train = torch.randn_like(X_train)
                X_val = torch.randn_like(X_val)
                X_test = torch.randn_like(X_test)

            if model_name == 'gcn':
                model = LP_GCN(data_train.num_features, 256, 512).to(device)
            elif model_name == 'hypergcn':
                model = LP_HyperGCN(data_train.num_features, 256, 512).to(device)
            best_loss = float('inf')
            best_model = None
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

            begin = time()
            for epoch in range(1000):
                model.train()
                optimizer.zero_grad()
                _, y = model(X_train.to(device), train_data.edge_index.to(device), edge_index)
                y = y @ y.t()
                loss = criterion(y[train_data.edge_label_index[0], train_data.edge_label_index[1]], train_data.edge_label.to(device))
                loss.backward()
                optimizer.step()
                y = torch.sigmoid(y)
                roc_auc = roc_auc_score(train_data.edge_label.cpu().detach().numpy(), y[train_data.edge_label_index[0], train_data.edge_label_index[1]].cpu().detach().numpy())
                history["train"]["loss"].append(loss.item())
                history["train"]["roc_auc"].append(roc_auc)
                model.eval()
                with torch.no_grad():
                    _, y = model(X_val.to(device), val_data.edge_index.to(device), edge_index)
                    y = y @ y.t()
                    val_loss = criterion(y[val_data.edge_label_index[0], val_data.edge_label_index[1]], val_data.edge_label.to(device))
                    y = torch.sigmoid(y)
                    val_roc_auc = roc_auc_score(val_data.edge_label.cpu().detach().numpy(), y[val_data.edge_label_index[0], val_data.edge_label_index[1]].cpu().detach().numpy())
                    history["val"]["loss"].append(val_loss.item())
                    history["val"]["roc_auc"].append(val_roc_auc)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = model.state_dict()
            end = time()
            elapsed = end - begin
            times.append(elapsed)
            plt.plot(history["train"]["loss"], label='Train Loss')
            plt.plot(history["val"]["loss"], label='Val Loss')
            plt.legend()
            plt.yscale('log')
            plt.savefig(f'plots/loss_{experiment}.png')
            plt.close()
            with torch.no_grad():
                model.load_state_dict(best_model)
                model.eval()
                _, y = model(X_test.to(device), test_data.edge_index.to(device), edge_index)
                y = y @ y.t()
                y = torch.sigmoid(y)
                roc_auc = roc_auc_score(test_data.edge_label.cpu().detach().numpy(), y[test_data.edge_label_index[0], test_data.edge_label_index[1]].cpu().detach().numpy())
                print(f'Time {elapsed} Test ROC AUC {roc_auc:.4f}')
                results.append(roc_auc)
        print(f'Average Test ROC AUC {np.mean(results):.4f}')
        print(f'Average Time {np.mean(times):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['gcn', 'hypergcn'], required=True)
    parser.add_argument('--random_features', action='store_true')
    parser.add_argument('--score_function', type=str, help='Score function to use', required=True, choices=['jc', 'aa', 'ra'])

    args = parser.parse_args()

    if args.score_function == 'jc':
        score_function = nx.jaccard_coefficient
    elif args.score_function == 'aa':
        score_function = nx.adamic_adar_index
    elif args.score_function == 'ra':
        score_function = nx.resource_allocation_index

    main(args.model, args.random_features, score_function)
