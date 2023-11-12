
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset


class MatrixFactorization(nn.Module):
    def __init__(self, *, n_users, n_items, n_dims=64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_dims = n_dims
        self.U = nn.Embedding(n_users, n_dims, sparse=False)
        self.V = nn.Embedding(n_items, n_dims, sparse=False)

    def forward(self, users=None, items=None):
        U = self.U.weight if users is None else self.U(users)
        V = self.V.weight if items is None else self.V(items)
        return (U @ V.T)

    def __repr__(self):
        n_users = self.n_users
        n_items = self.n_items
        n_dims = self.n_dims
        return f"<MatrixFactorization({n_users=}, {n_items=}, {n_dims=})>"


class ExpertNet(nn.Module):
    def __init__(self, u, V):
        super().__init__()
        if isinstance(u, np.ndarray):
            u = torch.Tensor(u)
        if isinstance(V, np.ndarray):
            V = torch.Tensor(V)
        self.u = nn.parameter.Parameter(u.view(-1,1))
        self.V = V
    
    def forward(self, x):
        n, d = x.shape
        logits = (self.V @ self.u).T
        return logits.expand(n, -1)
