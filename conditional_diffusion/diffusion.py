import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from conditional_diffusion.denoiser import Denoiser
from conditional_diffusion.logger import StepLogger


@dataclass(frozen=True)
class DiffusionSettings:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.75
    rho: float = 7.0
    p_mean: float = -1.2
    p_std: float = 1.2

    def sample_sigma(self, size: int) -> np.ndarray:
        ln_sigma = np.random.standard_normal(size) * self.p_std + self.p_mean
        return np.exp(ln_sigma)

    def t_grid(self, num_steps: int) -> np.ndarray:
        return (
            self.sigma_max ** (1 / self.rho)
            + np.arange(num_steps)
            / (num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho


@dataclass
class Preconditioning:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor

    @classmethod
    def of_sigma(cls, sigma: Tensor, sigma_data: float):
        sum_sq = sigma**2 + sigma_data**2
        sqrt_sum_sq = torch.sqrt(sum_sq)
        c_in = 1 / sqrt_sum_sq
        c_out = sigma * sigma_data / sqrt_sum_sq
        c_skip = sigma_data**2 / sum_sq
        c_noise = torch.log(sigma) / 4
        return cls(
            c_in=c_in,
            c_out=c_out,
            c_skip=c_skip,
            c_noise=c_noise,
        )

    def target(self, sample: Tensor, noised_sample: Tensor) -> Tensor:
        return (1 / self.c_out) * (sample - self.c_skip * noised_sample)

    def pred_target(self, noised_sample: Tensor, denoiser: Denoiser) -> Tensor:
        return denoiser.forward(self.c_in * noised_sample, self.c_noise)

    def pred(self, noised_sample: Tensor, pred_target: Tensor) -> Tensor:
        return self.c_skip * noised_sample + self.c_out * pred_target

    def pred_from_noised_sample(
        self, noised_sample: Tensor, denoiser: Denoiser
    ) -> Tensor:
        pred_target = self.pred_target(noised_sample, denoiser)
        return self.pred(noised_sample, pred_target)


@dataclass(frozen=True)
class BatchOutput:
    target: Tensor
    pred_target: Tensor
    pred: Tensor


@dataclass(frozen=True)
class DiffusionStatistics:
    loss: np.ndarray


def _to_torch(array: np.ndarray) -> Tensor:
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(1)


class Diffusion:
    def __init__(self, diffusion_settings: DiffusionSettings):
        self._settings = diffusion_settings
        self._denoiser = Denoiser()
        self._optimizer = torch.optim.Adam(self._denoiser.model.parameters(), lr=0.001)
        self._loss = nn.MSELoss()

    @classmethod
    def load(cls, model_filename: str, settings_filename: str):
        # Load settings
        with open(settings_filename, "r") as file:
            settings_data = json.load(file)
        settings = DiffusionSettings(**settings_data)
        # Instantiate class
        diffusion = cls(settings)
        # Load model weights
        diffusion._denoiser.model.load_state_dict(
            torch.load(model_filename, weights_only=True)
        )
        return diffusion

    def save(self, model_filename: str, settings_filename: str):
        # Save model
        torch.save(self._denoiser.model.state_dict(), model_filename)
        # Save settings
        settings_data = asdict(self._settings)
        with open(settings_filename, "w") as file:
            json.dump(settings_data, file, indent=4)

    def _preconditioning(self, sigma: Tensor) -> Preconditioning:
        return Preconditioning.of_sigma(sigma, self._settings.sigma_data)

    def pred(self, sigma: float, noised_sample: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            sigma_tensor = _to_torch(np.full_like(noised_sample, sigma))
            noised_sample_tensor = _to_torch(noised_sample)
            preconditioning = self._preconditioning(sigma_tensor)
            pred = preconditioning.pred_from_noised_sample(
                noised_sample_tensor, self._denoiser
            )
            pred_arr = pred.numpy().flatten()
        return pred_arr

    def _single_batch(self, batch: np.ndarray) -> BatchOutput:
        # Sample and noise
        sigma = self._settings.sample_sigma(len(batch))
        noise = np.random.standard_normal(len(batch)) * sigma

        # Numpy items as tensors
        sigma_tensor = _to_torch(sigma)
        sample_tensor = _to_torch(batch)
        noised_sample_tensor = _to_torch(batch + noise)

        # Precondition from sigma
        preconditioning = self._preconditioning(sigma_tensor)
        target = preconditioning.target(sample_tensor, noised_sample_tensor)
        pred_target = preconditioning.pred_target(noised_sample_tensor, self._denoiser)
        pred = preconditioning.pred(noised_sample_tensor, pred_target)

        return BatchOutput(target=target, pred_target=pred_target, pred=pred)

    def train_or_test_loop(
        self,
        batches: list[np.ndarray],
        train: bool,
        logger: Optional[StepLogger] = None,
    ) -> DiffusionStatistics:
        loss = np.zeros(len(batches))
        if train:
            self._denoiser.model.train()
        else:
            self._denoiser.model.eval()
        for i, batch in enumerate(batches):
            if train:
                batch_output = self._single_batch(batch)
                loss_current = self._loss(batch_output.target, batch_output.pred_target)
                loss_current.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
            else:
                with torch.no_grad():
                    batch_output = self._single_batch(batch)
                    loss_current = self._loss(
                        batch_output.target, batch_output.pred_target
                    )

            # Record loss and RMSE
            loss[i] = loss_current.item()
            if logger is not None:
                logger.update(i, f"Loss: {loss[i]}")

        return DiffusionStatistics(loss=loss)
