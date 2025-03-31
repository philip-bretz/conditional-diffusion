from dataclasses import dataclass

import numpy as np

from conditional_diffusion.diffusion import Diffusion


@dataclass(frozen=True)
class SamplerHyperparameters:
    tau: float
    grid_size: int = 25
    num_corrections: int = 2


@dataclass(frozen=True)
class Sample:
    array: np.ndarray

    def num_steps(self) -> int:
        return self.array.shape[0]

    def sample_size(self) -> int:
        return self.array.shape[1]

    def dim(self) -> int:
        return self.array.shape[2]

    def get_chain(self, index: int) -> np.ndarray:
        return self.array[:,index,:]

    def get_sample(self) -> np.ndarray:
        return self.array[-1,:,:]

    def get_sample_at_step(self, index: int) -> np.ndarray:
        return self.array[index,:,:]


class PredictorCorrectorSampler:
    def __init__(self, diffusion: Diffusion, hyper_params: SamplerHyperparameters):
        self._diffusion = diffusion
        self._hyper_params = hyper_params

    def t_grid(self) -> np.ndarray:
        return np.flip(np.linspace(0, 1, self._hyper_params.grid_size))

    def generate_sample(self, size: int) -> Sample:
        t_grid = self.t_grid()
        N = len(t_grid)
        dim = self._diffusion.dim()
        X = np.zeros((N, size, dim))
        X[0] = np.random.standard_normal((size, dim))
        for i in range(N - 1):
            x = self._exponential_integrator_step(X[i], t_grid[i], t_grid[i+1])
            for _ in range(self._hyper_params.num_corrections):
                x = self._langevin_correction(x, t_grid[i])
            X[i+1] = x
        return Sample(array=X)

    def _exponential_integrator_step(self, x: np.ndarray, t_1: float, t_2: float) -> np.ndarray:
        """
        Exponential integrator reversing the SDE

        See Eq. (16) in https://arxiv.org/pdf/2306.10574
        """
        score = self._diffusion.score(x, t_1)
        mu_1, sigma_1 = self._diffusion.var_schedule.mu_sigma_for_t(t_1)
        mu_2, sigma_2 = self._diffusion.var_schedule.mu_sigma_for_t(t_2)
        return mu_2 / mu_1 * x + (mu_2 / mu_1 - sigma_2 / sigma_1) * (sigma_1 ** 2) * score

    def _langevin_correction(self, x: np.ndarray, t: float):
        eps = np.random.standard_normal(x.shape)
        score = self._diffusion.score(x, t)
        delta = self._hyper_params.tau * self._diffusion.dim() / np.sum(score ** 2, axis=1)
        return x + delta * score + np.sqrt(2 * delta) * eps
