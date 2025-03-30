import torch
from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, x_size: int = 1, n_internal: int = 256, n_output: int = 128):
        super().__init__()
        self.hidden = nn.Linear(2 * x_size, n_internal)
        self.hidden2 = nn.Linear(n_internal, n_internal)
        self.hidden3 = nn.Linear(n_internal, n_output)
        self.output = nn.Linear(n_output, x_size)

    def forward(self, noised_x: Tensor, sigma: Tensor):
        x = torch.column_stack((noised_x, sigma))
        x = torch.relu(self.hidden(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x
