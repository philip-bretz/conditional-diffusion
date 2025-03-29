import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x_1, x_2):
        x = torch.column_stack((x_1, x_2))
        x = torch.relu(self.hidden(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x
