import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

X      = np.load("data/X_genotype.npy")
labels = np.load("data/labels.npy")
PRS    = np.load("data/PRS.npy")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

n_early = (labels == 0).sum(); n_late = (labels == 1).sum()
pos_weight = torch.tensor([n_late / n_early], dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),  nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, seed, epochs=200, lr=1e-3, hidden=128, dropout=0.3):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1], hidden=hidden, dropout=dropout)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xtr = torch.tensor(X_tr); ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(X_va); yva = torch.tensor(y_va, dtype=torch.float32)
    Xte = torch.tensor(X_te)

    best_val_auc, best_state = 0, None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xtr), ytr)
        loss.backward(); opt.step()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_prob = torch.sigmoid(model(Xva)).numpy()
            val_auc = roc_auc_score(y_va, val_prob)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_prob = torch.sigmoid(model(Xte)).numpy()
    te_pred = (te_prob >= 0.5).astype(int)
    return roc_auc_score(y_te, te_prob), f1_score(y_te, te_pred, average='macro')

class MLP3(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),  nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def run_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, seed, epochs=300, lr=1e-3, hidden=64, dropout=0.4):
    torch.manual_seed(seed)
    model = MLP3(X_tr.shape[1], hidden=hidden, dropout=dropout)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xtr = torch.tensor(X_tr); ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(X_va); Xte = torch.tensor(X_te)
    best_val, best_state = 0, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        crit(model(Xtr), ytr).backward(); opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(Xva)).numpy()
            va = roc_auc_score(y_va, vp)
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(Xte)).numpy()
    return roc_auc_score(y_te, tp), f1_score(y_te, (tp>=0.5).astype(int), average='macro')