
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

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        self.eval()
        dataloader = DataLoader(x, batch_size=100, shuffle=False)
        return torch.concat([ self.forward(x_batch) for x_batch in dataloader ])
