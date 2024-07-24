import os.path as osp
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

# Models used for link prediction
from models.lp.gcn import Model as LP_GCN
from models.lp.hypergcn import Model as LP_HyperGCN

