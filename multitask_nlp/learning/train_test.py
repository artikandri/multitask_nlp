from copy import copy
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.datasets.multitask_datamodule import MultiTaskDataModule
from multitask_nlp.learning.lightning_model import Model
from multitask_nlp.settings import CHECKPOINTS_DIR, LOGS_DIR


def load_model(
    model: nn.Module,
    ckpt_path=None,
):
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()

    return model


def train_test(
    datamodule: Union[BaseDataModule, MultiTaskDataModule],
    model: nn.Module,
    extra_test_datamodules: Optional[List[BaseDataModule]] = None,
    epochs: int = 6,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    use_cuda: bool = False,
    logger=None,
    log_model: Union[str, bool] = False,
    custom_callbacks: Optional[List[Callback]] = None,
    lightning_model_kwargs=None,
    trainer_kwargs=None,
    **kwargs
):
    """Performs training and testing of the model.

    The function uses Trainer Lightning API to perform training and testing. Training can be done
    using either a single task datamodule or a multitask datamodule.

    Args:
        datamodule (BaseDataModule or MultiTaskDataModule): Datamodule for training nad testing
        model (torch.nn.Module): Torch module which will be trained.
        extra_test_datamodules (list of BaseDataModule, optional): Additional tasks datamodule which
            will be only used for testing. Useful when we want to test other dataset which task type
            is the same, e.g., other POS tagging dataset when model was trained on another POS task.
        epochs (int, optional): Number of training epochs. Defaults to 6.
        lr (float, optional): Learning rate value. Defaults to 1e-2.
        weight_decay (float, optional): Weight decay regularization parameter. Defaults to 0.0 (no
            regularization).
        use_cuda (bool, optional): Whether to use GPU (CUDA) during training/testing or not.
            Defaults to False.
        logger (LightningLoggerBase, optional): Logger object used during training and testing.
            Defaults to None, then WANDB logger is created and used.
        log_model (str or bool, optional): Whether to log model or not. Parameter used only when no
            logger object is passed. See WandbLogger class documentation for details.
        custom_callbacks (list of Callback, optional): Custom Lightning callbacks used during
            training and testing. Defaults to None.
        lightning_model_kwargs (dict): A dictionary matching additional lightning model keyword
            parameters and their values. Defaults to None.
        trainer_kwargs (dict): A dictionary matching additional Trainer object keyword
            parameters and their values, e.g., accumulate_grad_batches. Defaults to None.
        **kwargs: Arbitrary keyword arguments. Currently, not used.
    """
    if extra_test_datamodules is None:
        extra_test_datamodules = []

    if isinstance(datamodule, MultiTaskDataModule):
        tasks_datamodules = datamodule.tasks_datamodules
    else:
        tasks_datamodules = [datamodule]

    lightning_model_kwargs = lightning_model_kwargs or {}
    lightning_model = Model(
        model=model,
        tasks_datamodules=tasks_datamodules,
        extra_test_datamodules=extra_test_datamodules,
        lr=lr,
        weight_decay=weight_decay,
        **lightning_model_kwargs
    )
    if logger is None:
        logger = pl_loggers.WandbLogger(save_dir=LOGS_DIR, log_model=log_model)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    if custom_callbacks is not None:
        callbacks = copy(custom_callbacks)
    else:
        callbacks = []

    # When none of the callbacks is ModelCheckpoint, a new default one is added which saves the
    # model with the minimum valid loss.
    if not any(isinstance(callback, ModelCheckpoint) for callback in callbacks):
        checkpoint_dir = CHECKPOINTS_DIR / logger.experiment.name
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=1,
                monitor='valid_loss',
                mode='min',
            )
        )
    _use_cuda = use_cuda and torch.cuda.is_available()

    trainer_kwargs = trainer_kwargs or {}
    trainer = pl.Trainer(
        gpus=1 if _use_cuda else 0,
        max_epochs=epochs,
        progress_bar_refresh_rate=20,
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=2,
        **trainer_kwargs
    )
    trainer.fit(lightning_model, train_loader, val_loader)

    test_dataloaders = [test_loader] + [extra_test_data_module.whole_dataset_dataloader() for
                                        extra_test_data_module in extra_test_datamodules]
    trainer.test(dataloaders=test_dataloaders)
