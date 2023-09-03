
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader


class MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_layers=1, hidden_dim=16):
        super().__init__()
        if hidden_layers == 0:
            self.net = self.net = nn.Linear(input_dim, output_dim)
        else:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ]
            for _ in range(hidden_layers):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ])
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def train(dataset: Dataset,
          model: nn.Module,
          epochs: int = 10,
          loss_fn: nn.Module = None,
          optimizer: torch.optim.Optimizer = None,
          device = "cpu") -> None:

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    size = len(dataloader.dataset)
    model.train()
    
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{epoch=}] [{current:>5d}/{size:>5d}]")        


def predict(dataset, model: nn.Module, device="cpu") -> tuple[Tensor, Tensor]:
    dataloader = DataLoader(dataset, batch_size=100)
    model.eval()
    ys = []
    yhats = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            ys.append(y)
            yhats.append(model(X))
    return torch.concat(ys), torch.concat(yhats)
