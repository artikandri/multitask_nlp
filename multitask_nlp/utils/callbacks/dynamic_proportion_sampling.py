"""Module with callbacks which control smoothing parameters of proportional sampling of tasks."""
from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl

from multitask_nlp.datasets import ProportionalSamplingMTDataset


class DynamicSampling(pl.Callback, ABC):
    """Abstract callback for determining smoothing parameter.

    Children class has to override method which gives exact value of alpha.
    """
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Sets new alpha parameter for Proprtional Sampling dataset.

        It is run at the beginning of each epoch. New value depends on epoch number.
        """
        current_epoch = trainer.fit_loop.current_epoch + 1
        mtl_dataset = trainer.train_dataloader.loaders.dataset
        if isinstance(mtl_dataset, ProportionalSamplingMTDataset):
            current_alpha = self._get_current_alpha(current_epoch)
            mtl_dataset.set_alpha(current_alpha)

    @abstractmethod
    def _get_current_alpha(self, current_epoch: int) -> float:
        """Abstract method which calculates a new alpha value.

        Args:
            current_epoch (int): Current epoch number. In that case numbered from 1.
        """
        pass


class AnnealingSampling(DynamicSampling):
    """Callback implementing annealing sampling proposed in "Bert and pals: Projected attention
     layers for efficient adaptation in multi-task learning" by Stickland et al. (2019).

     Attributes:
         epochs_number (int): Total number of epochs.
         beta (float, optional): Beta parameter. Defaults to 0.8.
    """
    def __init__(self, epochs_number: int, beta: float = 0.8) -> None:
        super().__init__()
        self.epochs_number = epochs_number
        self.beta = beta

    def _get_current_alpha(self, current_epoch: int) -> float:
        current_alpha = 1 - self.beta * (current_epoch - 1) / (self.epochs_number - 1)
        return current_alpha


class DynamicTemperatureSampling(DynamicSampling):
    """Callback for dynamic temperature proposed in "Multi-task Learning for Multilingual Neural
    Machine Translation" by Wang et al. (2020).

    In the paper, temperature parameter was used. In the project alpha parameter is used, which
    is an inversion of temperature: alpha = 1/T.

    Attributes:
        N (int): Number of warm-up epochs.
        T_0 (int, optional): Initial temperature value. Defaults to 1.
        T_max (int, optional): Maximum temperature value. Defaults to 5.
    """
    def __init__(self, N: int, T_0: int = 1, T_max: int = 5) -> None:
        super().__init__()
        assert T_0 > 0
        assert T_max > 0 and T_max > T_0
        assert N > 0
        self.N = N
        self.T_0 = T_0
        self.T_max = T_max

    def _get_current_alpha(self, current_epoch: int) -> float:
        current_temperature = np.minimum(
            self.T_max, (current_epoch - 1) * (self.T_max - self.T_0) / self.N + self.T_0
        )
        current_alpha = 1 / current_temperature
        return current_alpha
