import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neighbors import kneighbors_graph
import numpy as np

def build_knn_graph(dist_mat, k):
    N = dist_mat.shape[0]
    src, dst = [], []
    for i in range(N):
        nbrs = np.argsort(dist_mat[i])[1:k+1]
        for j in nbrs:
            src.append(i); dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden//2)
        self.conv3 = GCNConv(hidden//2, hidden//4)
        self.fc    = nn.Linear(hidden//4, 1)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x, ei):
        x = F.relu(self.conv1(x, ei)); x = self.drop(x)
        x = F.relu(self.conv2(x, ei)); x = self.drop(x)
        x = F.relu(self.conv3(x, ei))
        return self.fc(x).squeeze(-1)

class GAT(nn.Module):
    def __init__(self, in_dim, hidden=32, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden*heads, hidden, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden*heads, hidden, heads=1, concat=False, dropout=dropout)
        self.fc    = nn.Linear(hidden, 1)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x, ei):
        x = F.elu(self.conv1(x, ei)); x = self.drop(x)
        x = F.elu(self.conv2(x, ei)); x = self.drop(x)
        x = F.elu(self.conv3(x, ei))
        return self.fc(x).squeeze(-1)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden//2)
        self.conv3 = SAGEConv(hidden//2, hidden//4)
        self.fc    = nn.Linear(hidden//4, 1)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x, ei):
        x = F.relu(self.conv1(x, ei)); x = self.drop(x)
        x = F.relu(self.conv2(x, ei)); x = self.drop(x)
        x = F.relu(self.conv3(x, ei))
        return self.fc(x).squeeze(-1)

def build_graph(X_feat, labels, k):
    A = kneighbors_graph(X_feat, n_neighbors=k, metric='cosine',
                         mode='connectivity', include_self=False)
    A_sym = (A + A.T); A_sym.data[:] = 1.0
    edge_index, _ = from_scipy_sparse_matrix(A_sym)
    data = Data(
        x=torch.tensor(X_feat),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long)
    )
    return data

def train_gnn(model, feat, edge_index, labels, tr_idx, va_idx, te_idx,
              epochs=400, lr=1e-3, wd=1e-4, cw=6.47):
    x_t  = torch.tensor(feat, dtype=torch.float)
    y_t  = torch.tensor(labels, dtype=torch.float)
    pw   = torch.tensor([cw])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_m = torch.zeros(len(labels), dtype=torch.bool); tr_m[tr_idx] = True
    va_m = torch.zeros(len(labels), dtype=torch.bool); va_m[va_idx] = True
    te_m = torch.zeros(len(labels), dtype=torch.bool); te_m[te_idx] = True

    best_va, best_state, patience = 0.0, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(x_t, edge_index)
        loss   = crit(logits[tr_m], y_t[tr_m])
        loss.backward(); opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            probs_va = torch.sigmoid(model(x_t, edge_index)[va_m]).numpy()
        va_auc = roc_auc_score(labels[va_idx], probs_va)
        if va_auc > best_va:
            best_va = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 50: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs_te = torch.sigmoid(model(x_t, edge_index)[te_m]).numpy()
    te_auc = roc_auc_score(labels[te_idx], probs_te)
    te_f1  = f1_score(labels[te_idx], (probs_te>=0.5).astype(int),
                      average='macro', zero_division=0)
    return te_auc, te_f1, probs_te

def quick_eval(feat, k=10, seed=42, model_cls=GCN, epochs=400, lr=5e-4, hidden=64):
    A = kneighbors_graph(X_scaled, n_neighbors=k, metric='cosine',
                         mode='connectivity', include_self=False)
    A_sym = (A + A.T); A_sym.data[:] = 1.0
    edge_index, _ = from_scipy_sparse_matrix(A_sym)
    data = Data(x=torch.tensor(feat), edge_index=edge_index,
                y=torch.tensor(labels, dtype=torch.long))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, te_idx = next(sss.split(feat, labels))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=seed)
    tr_idx2, va_idx = next(sss2.split(feat[tr_idx], labels[tr_idx]))
    tr_idx_final = tr_idx[tr_idx2]; va_idx_final = tr_idx[va_idx]
    tr_mask = torch.zeros(len(labels), dtype=torch.bool); tr_mask[tr_idx_final] = True
    va_mask = torch.zeros(len(labels), dtype=torch.bool); va_mask[va_idx_final] = True
    te_mask = torch.zeros(len(labels), dtype=torch.bool); te_mask[te_idx] = True

    torch.manual_seed(seed)
    in_d = feat.shape[1]
    model = model_cls(in_d, hidden=hidden, dropout=0.4) if model_cls==GCN else model_cls(in_d, hidden=32, heads=2, dropout=0.4)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val, best_state = 0, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        crit(model(data.x, data.edge_index)[tr_mask],
             data.y[tr_mask].float()).backward(); opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(data.x, data.edge_index)[va_mask]).numpy()
            va = roc_auc_score(data.y[va_mask].numpy(), vp)
            if va > best_val:
                best_val = va
                best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(data.x, data.edge_index)[te_mask]).numpy()
    return roc_auc_score(data.y[te_mask].numpy(), tp), best_val

def build_and_train(labels, pos_weight, from_scipy_sparse_matrix, X_scaled, model, feat, k, tr_m, va_m, seed, epochs=400):
    A = kneighbors_graph(X_scaled, n_neighbors=k, metric='cosine', mode='connectivity', include_self=False)
    A_sym = (A + A.T); A_sym.data[:] = 1.0
    ei, _ = from_scipy_sparse_matrix(A_sym)
    data = torch.utils.data.Dataset  # placeholder
    data = Data(x=torch.tensor(feat), edge_index=ei, y=torch.tensor(labels, dtype=torch.long))
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val, best_state = 0, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        crit(model(data.x, data.edge_index)[tr_m], data.y[tr_m].float()).backward(); opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(data.x, data.edge_index)[va_m]).numpy()
            va = roc_auc_score(data.y[va_m].numpy(), vp)
            if va > best_val:
                best_val = va; best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(data.x, data.edge_index)[te_m]).numpy()
    return tp, data, model