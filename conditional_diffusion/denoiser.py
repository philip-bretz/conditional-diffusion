import torch
from torch import Tensor
import numpy as np

from conditional_diffusion.models import MLP


def to_torch(array: np.ndarray) -> Tensor:
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(1)


class Denoiser:
    def __init__(self, x_size: int = 1, **model_kwargs):
        self.dim = x_size
        self.model = MLP(in_size=x_size + 1, out_size=x_size, **model_kwargs)

    def forward_numpy(self, x: np.ndarray, t: np.ndarray) -> Tensor:
        return self.forward(to_torch(x), to_torch(t))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = torch.column_stack((x, t))
        return self.model(x)
