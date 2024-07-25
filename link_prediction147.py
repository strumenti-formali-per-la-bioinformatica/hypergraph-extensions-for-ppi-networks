import torch
import numpy as np
from time import time
import networkx as nx
from os import listdir
import torch_geometric
import torch_geometric.data
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from models.lp.hypergcn import Model as LP_HyperGCN
from models.lp.gcn import Model as LP_GCN
from itertools import combinations
import matplotlib.pyplot as plt
import argparse

def main(model_name: str):
    remove_duplicated = T.RemoveDuplicatedEdges()
    transform = T.RandomLinkSplit(is_undirected=True, num_val=0.25, num_test=0.25)

    score_function = nx.jaccard_coefficient

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    x = torch.tensor(np.load("data/PPI147/ppi/ppi-feats.npy").astype(np.float32))
    for file_name in sorted(listdir("data/PPI147/bio-tissue-networks")):
        G = nx.read_edgelist("data/PPI147/bio-tissue-networks/" + file_name)
        edge_index = torch.tensor(np.array(nx.to_scipy_sparse_array(G).todense().nonzero()))
        data = torch_geometric.data.Data(x=x, edge_index=edge_index)
        data = remove_duplicated(data)
        for experiment in range(5):
            train_data, val_data, test_data = transform(data)
            G = nx.from_edgelist(train_data.edge_index.t().tolist())
            cliques = list(nx.find_cliques(G))
            hyperedges = cliques
            scores = []
            total_jc = 0
            for clique in cliques:
                jc = sum(map(lambda x: x[2], score_function(G, list(combinations(clique, 2))))) / len(clique)
                scores.append(jc)
                total_jc += jc
            avg_jc = total_jc / len(list(cliques))
            hyperedges = [clique for jc, clique in zip(scores, cliques) if jc > avg_jc]

            edge_index = torch.tensor([
                [n for e in hyperedges for n in e ],
                [i for i, e in enumerate(hyperedges) for n in e]
            ])
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
            if model_name == 'gcn':
                model = LP_GCN(train_data.num_features, 128, 256)
            elif model_name == 'hypergcn':
                model = LP_HyperGCN(train_data.num_features, 128, 256)
            model.to(device)

            best_loss = float('inf')
            best_model = None
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

            begin = time()
            for epoch in range(25 if model_name == 'gcn' else 10):
                model.train()
                optimizer.zero_grad()
                _, y = model(train_data.x.to(device), train_data.edge_index.to(device), edge_index.to(device))
                y = y.to("cpu")
                y = y @ y.t()
                loss = criterion(y[train_data.edge_label_index[0], train_data.edge_label_index[1]], train_data.edge_label)
                loss.backward()
                optimizer.step()
                y = torch.sigmoid(y)
                roc_auc = roc_auc_score(train_data.edge_label.cpu().detach().numpy(), y[train_data.edge_label_index[0], train_data.edge_label_index[1]].cpu().detach().numpy())
                history["train"]["loss"].append(loss.item())
                history["train"]["roc_auc"].append(roc_auc)
                model.eval()
                with torch.no_grad():
                    _, y = model(val_data.x.to(device), val_data.edge_index.to(device), edge_index.to(device))
                    y = y.to("cpu")
                    y = y @ y.t()
                    val_loss = criterion(y[val_data.edge_label_index[0], val_data.edge_label_index[1]], val_data.edge_label)
                    y = torch.sigmoid(y)
                    val_roc_auc = roc_auc_score(val_data.edge_label.cpu().detach().numpy(), y[val_data.edge_label_index[0], val_data.edge_label_index[1]].cpu().detach().numpy())
                    history["val"]["loss"].append(val_loss.item())
                    history["val"]["roc_auc"].append(val_roc_auc)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = model.state_dict()
                print(f'Epoch {epoch} Train Loss {loss:.4f} Train ROC AUC {roc_auc:.4f} Val Loss {val_loss:.4f} Val ROC AUC {val_roc_auc:.4f}')
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
                _, y = model(test_data.x.to(device), test_data.edge_index.to(device), edge_index.to(device))
                y = y.to("cpu")
                y = y @ y.t()
                y = torch.sigmoid(y)
                roc_auc = roc_auc_score(test_data.edge_label.cpu().detach().numpy(), y[test_data.edge_label_index[0], test_data.edge_label_index[1]].cpu().detach().numpy())
                print(f'Time {elapsed} Test ROC AUC {roc_auc:.4f}')
                results.append(roc_auc)
        print(f'Average Test ROC AUC {np.mean(results):.4f}')
        print(f'Average Time {np.mean(times):.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hypergcn', help='Model to use', required=True, choices=['gcn', 'hypergcn'])
    args = parser.parse_args()
    model_name = args.model
    main(model_name)
