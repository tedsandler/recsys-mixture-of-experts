
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader


class MatrixFactorization(nn.Module):
    def __init__(self, n_user, n_item, n_dim=64):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_dim = n_dim
        self.U = nn.Embedding(n_user, n_dim, sparse=False)
        self.V = nn.Embedding(n_item, n_dim, sparse=False)

    def forward(self, users=None, items=None):
        U = self.U.weight if users is None else self.U(users)
        V = self.V.weight if items is None else self.V(items)
        return (U @ V.T)

    def __repr__(self):
        n_user = self.n_user
        n_item = self.n_item
        n_dim = self.n_dim
        return f"<MatrixFactorization({n_user=}, {n_item=}, {n_dim=})>"


def train(Y: Tensor,
          model: nn.Module,
          epochs: int = 10,
          batch_size = 32,
          loss_fn: nn.Module = None,
          optimizer: torch.optim.Optimizer = None,
          device="cpu") -> None:

    Y = Y.to(device)
    model = model.to(device)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)

    size = model.n_user
    dataloader = DataLoader(torch.arange(size), batch_size=batch_size, shuffle=True)
    model.train()
    
    for epoch in range(epochs):
        batch_loss = 0
        n_batch = 0
        for batch, batch_users in enumerate(dataloader):
            # Compute prediction error
            logits = model(batch_users)
            y = Y[batch_users, :]
            loss = loss_fn(logits, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_loss += loss.item()
            n_batch += 1
        if epoch % 10 == 0:
            print(f"loss: {batch_loss:>7f} [{epoch=}]") 
    print(f"loss: {batch_loss:>7f} [{epoch=}]") 
    return
