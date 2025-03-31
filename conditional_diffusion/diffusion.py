import json
from dataclasses import asdict, dataclass
from typing import Optional, TypeVar

import numpy as np
import torch
from torch import Tensor, nn

from conditional_diffusion.denoiser import Denoiser, to_torch
from conditional_diffusion.logger import StepLogger


DEFAULT_OMEGA = float(np.arccos(10 ** (-3)))
DEFAULT_GAMMA = 10 ** (-2)
T = TypeVar("T", float, np.ndarray)


@dataclass(frozen=True)
class DiffusionHyperParameters:
    dim: int = 1
    omega: float = DEFAULT_OMEGA
    learning_rate: float = 0.001


@dataclass(frozen=True)
class NoisedBatch:
    x: np.ndarray
    eps: np.ndarray


@dataclass(frozen=True)
class CosineVarianceSchedule:
    omega: float = DEFAULT_OMEGA

    def mu_sigma_for_t(self, t: T) -> tuple[T, T]:
        mu = np.cos(self.omega * t) ** 2
        sigma = np.sqrt(1 - mu ** 2)
        return mu, sigma

    def score(self, eps: np.ndarray, t: float) -> np.ndarray:
        """
        Re-parameterized score for noise at a specified level t

        See p. 3 in https://arxiv.org/pdf/2306.10574 for eps = -sigma * score parameterization
        """
        _, sigma = self.mu_sigma_for_t(t)
        return eps / -sigma

    def noise_batch(self, batch: np.ndarray, t: np.ndarray) -> NoisedBatch:
        """
        Add noise to sample at specified level t in [0, 1]

        Noise is added by calculating mu(t), sigma(t) and then applying Algorithm 1 in 
        https://arxiv.org/pdf/2306.10574: x <- mu * x + sigma * eps where eps is Gaussian
        """
        assert len(batch.shape) == 2
        assert t.shape == (batch.shape[0],)
        eps = np.random.standard_normal(batch.shape)
        mu, sigma = self.mu_sigma_for_t(t)
        x = mu[:,np.newaxis] * batch + sigma[:,np.newaxis] * eps
        return NoisedBatch(x=x, eps=eps)


@dataclass(frozen=True)
class BatchOutput:
    eps: Tensor
    pred_eps: Tensor    


class Diffusion:
    def __init__(self, hyper_params: DiffusionHyperParameters):
        self._hyper_params = hyper_params
        self.var_schedule = CosineVarianceSchedule(self._hyper_params.omega)
        self._denoiser = Denoiser(x_size=self._hyper_params.dim)
        self._optimizer = torch.optim.Adam(self._denoiser.model.parameters(), lr=self._hyper_params.learning_rate)
        self._loss = nn.MSELoss()

    @classmethod
    def load(cls, model_filename: str, hyper_params_filename: str):
        # Load settings
        with open(hyper_params_filename, "r") as file:
            hyper_params_data = json.load(file)
        hyper_params = DiffusionHyperParameters(**hyper_params_data)
        # Instantiate class
        diffusion = cls(hyper_params)
        # Load model weights
        diffusion._denoiser.model.load_state_dict(
            torch.load(model_filename, weights_only=True)
        )
        return diffusion

    def dim(self) -> int:
        return self._denoiser.dim

    def save(self, model_filename: str, hyper_params_filename: str):
        # Save model
        torch.save(self._denoiser.model.state_dict(), model_filename)
        # Save settings
        hyper_params_data = asdict(self._hyper_params)
        with open(hyper_params_filename, "w") as file:
            json.dump(hyper_params_data, file, indent=4)

    def pred(self, noised_sample: np.ndarray, t: float) -> np.ndarray:
        assert len(noised_sample.shape) == 2
        assert noised_sample.shape[1] == self.dim()
        with torch.no_grad():
            t_arr = np.full(noised_sample.shape[0], t)
            pred_noise = self._denoiser.forward_numpy(noised_sample, t_arr).numpy().flatten()
        pred_sample = noised_sample - pred_noise
        return pred_sample

    def score(self, noised_sample: np.ndarray, t: float) -> np.ndarray:
        pred_eps = self.pred(noised_sample, t)
        return self.var_schedule.score(pred_eps, t)

    def _single_batch(self, batch: np.ndarray) -> BatchOutput:
        if len(batch.shape) == 1:
            batch = batch[:,np.newaxis]
        assert len(batch.shape) == 2
        assert batch.shape[1] == self.dim()
        t = np.random.uniform(size=len(batch))
        noised_sample = self.var_schedule.noise_batch(batch, t)
        pred_eps = self._denoiser.forward_numpy(noised_sample.x, t)
        return BatchOutput(eps=to_torch(noised_sample.eps), pred_eps=pred_eps)

    def train_or_test_loop(
        self,
        batches: list[np.ndarray],
        train: bool,
        logger: Optional[StepLogger] = None,
    ) -> np.ndarray:
        loss = np.zeros(len(batches))
        if train:
            self._denoiser.model.train()
        else:
            self._denoiser.model.eval()
        for i, batch in enumerate(batches):
            if train:
                batch_output = self._single_batch(batch)
                loss_current = self._loss(batch_output.eps, batch_output.pred_eps)
                loss_current.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
            else:
                with torch.no_grad():
                    batch_output = self._single_batch(batch)
                    loss_current = self._loss(
                        batch_output.eps, batch_output.pred_eps
                    )

            # Record loss and RMSE
            loss[i] = loss_current.item()
            if logger is not None:
                logger.update(i, f"Loss: {loss[i]}")

        return loss
