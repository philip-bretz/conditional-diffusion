import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, n_internal: int = 256, n_output: int = 128
    ):
        super().__init__()
        self.hidden = nn.Linear(in_size, n_internal)
        self.hidden2 = nn.Linear(n_internal, n_internal)
        self.hidden3 = nn.Linear(n_internal, n_output)
        self.output = nn.Linear(n_output, out_size)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.hidden(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x
