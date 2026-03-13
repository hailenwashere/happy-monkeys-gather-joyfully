# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates


class LSTMCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 3,
        lr: float = 1e-3,
        input_size: int = None,
        decoder: DictConfig = None,
        **kwargs,  # absorb extra Hydra args
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size or (self.NUM_BANDS * self.ELECTRODE_CHANNELS)

        # Model
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,  # expects (T, N, input_size)
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, charset().num_classes)

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder — instantiate from DictConfig if passed via Hydra
        if isinstance(decoder, DictConfig):
            self.decoder = instantiate(decoder)
        else:
            self.decoder = decoder

        # Metrics — match TDSConvCTCModule structure exactly so val/CER is logged
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        T, N = inputs.shape[:2]
        x = inputs.view(T, N, -1)  # (T, N, input_size)
        outputs, _ = self.lstm(x)   # (T, N, hidden_size * 2)
        logits = self.classifier(outputs)  # (T, N, num_classes)
        return logits.log_softmax(dim=-1)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # Account for any temporal shrinkage from the model
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            emissions,                      # (T, N, num_classes)
            targets.transpose(0, 1),        # (T, N) -> (N, T)
            emission_lengths,               # (N,)
            target_lengths,                 # (N,)
        )

        # Decode and update CER metrics
        if self.decoder is not None:
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )
            metrics = self.metrics[f"{phase}_metrics"]
            targets_np = targets.detach().cpu().numpy()
            target_lengths_np = target_lengths.detach().cpu().numpy()
            for i in range(N):
                target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
                metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")  # logs val/CER for ModelCheckpoint

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
