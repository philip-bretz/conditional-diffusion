from dataclasses import dataclass

import numpy as np

from conditional_diffusion.diffusion import Diffusion


@dataclass(frozen=True)
class SamplerSettings:
    s_noise: float = 1.0
    s_churn: float = 1.0
    num_steps: int = 25

    def gamma(self) -> np.ndarray:
        return np.full(
            self.num_steps, min(self.s_churn / self.num_steps, np.sqrt(2) - 1)
        )


class StochasticSampler:
    def __init__(self, diffusion: Diffusion, sampler_settings: SamplerSettings):
        self._diffusion = diffusion
        self._settings = sampler_settings
        self._t_grid = self._diffusion._settings.t_grid(sampler_settings.num_steps)
        self._gamma = self._settings.gamma()

    def generate_sample(self, size: int, return_full: bool = False) -> np.ndarray:
        x = np.random.standard_normal(size) * self._t_grid[0]
        if return_full:
            X = np.zeros((len(self._t_grid), size))
            X[0] = x[:]
        for i in range(len(self._t_grid) - 1):
            eps = np.random.standard_normal(size) * self._settings.s_noise
            t = self._t_grid[i]
            t_hat = t * (1 + self._gamma[i])
            x_hat = x + np.sqrt(t_hat**2 - t**2) * eps
            d = (x_hat - self._diffusion.pred(t_hat, x_hat)) / t_hat
            t_new = self._t_grid[i + 1]
            x_new = x_hat + (t_new - t_hat) * d
            if t_new > 0:
                d_dash = (x_new - self._diffusion.pred(t_new, x_new)) / t_new
                x_new = x_hat + (t_new - t_hat) * (0.5 * d + 0.5 * d_dash)
            x = x_new
            if return_full:
                X[i + 1] = x[:]
        if return_full:
            return X
        return x
