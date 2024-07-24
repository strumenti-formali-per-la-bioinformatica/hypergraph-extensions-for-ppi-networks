import os.path as osp
from time import time

import torch

from torch_geometric.datasets import PPI

# Models
# Models used for function prediction
from models.fp.gcn import Model as FP_GCN
from models.fp.hypergcn import Model as FP_HyperGCN

import argparse
import logging

import pickle
from utils import plot_results
import numpy as np

from sklearn.metrics import roc_auc_score

def main(model_name: str):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PPI')
    data_train = PPI(root=path, split='train')
    data_val = PPI(root=path, split='val')
    data_test = PPI(root=path, split='test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if model_name == 'gcn':
        model = FP_GCN(data_train.num_features, 128, data_train.num_classes)
    elif model_name == 'hypergcn':
        model = FP_HyperGCN(data_train.num_features, 128, data_train.num_classes)
    model = model.to(device)

    history = {
        'train': {
            'loss': [],
            'auc': []
        },
        'val': {
            'loss': [],
            'auc': []
        }
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025)
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = 5000
    patience = 20
    print(f'Patience: {patience}')
    best_epoch = None
    best_val_loss = None
    best_val_auc = None
    best_model = None
    window = 100
    moving_losses = [0] * window
    moving_aucs = [0] * window
    loss_moving_average = np.float64('inf')
    auc_moving_average = 0
    print('Start training')

    train_edge_index = torch.tensor(pickle.load(open('data/train_edge_index_aa.pkl', 'rb')))
    val_edge_index = torch.tensor(pickle.load(open('data/val_edge_index_aa.pkl', 'rb')))
    test_edge_index = torch.tensor(pickle.load(open('data/test_edge_index_aa.pkl', 'rb')))

    X_train, edge_index_train, y_train = data_train.x, data_train.edge_index, data_train.y
    X_val, edge_index_val, y_val = data_val.x, data_val.edge_index, data_val.y
    X_test, edge_index_test, y_test = data_test.x, data_test.edge_index, data_test.y

    try:
        begin = time()
        for epoch in range(epochs):
            data_train = data_train.shuffle()
            model.train()
            optimizer.zero_grad()
            embs, out = model(X_train.to(device), edge_index_train.to(device), train_edge_index.to(device))
            loss = criterion(out, y_train.to(device))
            loss.backward()
            optimizer.step()
            out_train = torch.sigmoid(out).cpu().detach().numpy()
            roc_auc_train = roc_auc_score(y_train.cpu().detach().numpy(), out_train)
            history['train']['loss'].append(loss.item())
            history['train']['auc'].append(roc_auc_train)

            model.eval()
            with torch.no_grad():
                embs, out = model(X_val.to(device), edge_index_val.to(device), val_edge_index.to(device))
                val_loss = criterion(out, y_val.to(device))
                out_val = torch.sigmoid(out).cpu().detach().numpy()
                roc_auc_val = roc_auc_score(y_val.cpu().detach().numpy(), out_val)
                history['val']['loss'].append(val_loss.item())
                history['val']['auc'].append(roc_auc_val)

            moving_losses[epoch % window] = val_loss.item()
            moving_aucs[epoch % window] = roc_auc_val

            if best_val_auc is None or roc_auc_val > best_val_auc:
                best_val_auc = roc_auc_val
                best_epoch = epoch
                best_model = model.state_dict()

            if epoch - best_epoch >= patience:
                break

            if epoch % 5 == 0:
                logging.info(f'Epoch: {epoch}, Train loss: {loss.item():.4f}, Val loss: {val_loss.item():.4f} {epoch - best_epoch}/{patience}, Train AUC: {roc_auc_train:.4f}, Val AUC: {roc_auc_val:.4f} {best_val_auc:.4f}')

    except KeyboardInterrupt:
        logging.info('Interrupted by user')
        plot_results(history)

    end = time()
    elapsed = end - begin
    logging.info(f'Training time: {elapsed}')

    plot_results(history)

    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        embs, out = model(X_test.to(device), edge_index_test.to(device), test_edge_index.to(device))
        test_loss = criterion(out, y_test.to(device))
        test_pred = torch.sigmoid(out).cpu().detach().numpy()
        test_true = y_test.cpu().detach().numpy()
        auc = roc_auc_score(test_true, test_pred)
        logging.info(f'Test loss: {test_loss.item()}, Test AUC: {auc}')
        print(f'Test loss: {test_loss.item()}, Elapsed Time {elapsed}, Test AUC: {auc}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Function Prediction')
    parser.add_argument('--model', type=str, default='gcn', help='Model to use', choices=['gcn', 'hypergcn'], required=True)
    
    args = parser.parse_args()

    model_name = args.model

    logging.basicConfig(level=logging.INFO)

    main(model_name)

