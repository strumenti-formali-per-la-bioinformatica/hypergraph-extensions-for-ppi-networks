import os.path as osp
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

# Models
# Models used for function prediction
from models.fp.gcn import Model as FP_GCN
from models.fp.hypergcn import Model as FP_HyperGCN

import pickle

def main():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')

if __name__ == "__main__":
    main()

