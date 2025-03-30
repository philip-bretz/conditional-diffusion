import torch
from torch import Tensor

from conditional_diffusion.models import MLP

class Denoiser:
    def __init__(self, x_size: int = 1, **model_kwargs):
        self.model = MLP(in_size=x_size * 2, out_size=x_size, **model_kwargs)
    
    def forward(self, noised_x: Tensor, sigma: Tensor) -> Tensor:
        x = torch.column_stack((noised_x, sigma))
        return self.model(x)
