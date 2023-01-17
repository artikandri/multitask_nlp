import os
from copy import copy
from itertools import product
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from multitask_nlp.datasets.studemo.studemo import StudEmoDataModule
from multitask_nlp.learning.train_test import train_test
from multitask_nlp.models import models as models_dict
from multitask_nlp.settings import CHECKPOINTS_DIR, LOGS_DIR
from multitask_nlp.utils import seed_everything

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_START_METHOD"] = "thread"

RANDOM_SEED = 42

if __name__ == "__main__":
    datamodule_cls = StudEmoDataModule
    rep_num = 1

    model_types = ['multitask_transformer']
    model_names = ['bert']
    wandb_project_name = 'StudEmo_SingleExp_EarlyStopping'

    max_length = 256
    lr_rate = 1e-5
    epochs = 10
    batch_size = 32
    weight_decay = 0.1
    warmup_proportion = 0.06

    use_cuda = False
    custom_callbacks: List[pl.Callback] = [
        LearningRateMonitor()
    ]
    lightning_model_kwargs = {
        'lr_scheduling': True,
        'warmup_proportion': warmup_proportion
    }

    seed_everything()
    np.random.seed(RANDOM_SEED)

    data_module = datamodule_cls(batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup()

    for model_type, model_name, in product(model_types, model_names):
        model_cls = models_dict[model_type]
        hparams = {
            "learning_kind": 'STL',
            "dataset": type(data_module).__name__,
            "model_type": model_type,
            "model_name": model_name,
            "num_epochs": epochs,
            "learning_rate": lr_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "warmup_proportion": warmup_proportion,
            "max_length": max_length
        }
        for i in range(rep_num):
            torch.manual_seed(RANDOM_SEED + i)
            model = model_cls(
                tasks_datamodules=data_module,
                model_name=model_name,
                max_length=max_length
            )

            logger = pl_loggers.WandbLogger(
                save_dir=str(LOGS_DIR),
                config=hparams,
                project=wandb_project_name,
                log_model=False,
            )

            exp_custom_callbacks = copy(custom_callbacks)
            exp_custom_callbacks.extend(
                [
                    ModelCheckpoint(
                        dirpath=CHECKPOINTS_DIR / logger.experiment.name,
                        save_top_k=1,
                        monitor='valid_overall_score',
                        mode='max',
                    ),
                    EarlyStopping(
                        monitor='valid_overall_score',
                        patience=5,
                        mode='max'
                    )
                ]
            )

            train_test(
                datamodule=data_module,
                model=model,
                epochs=epochs,
                lr=lr_rate,
                weight_decay=weight_decay,
                use_cuda=use_cuda,
                logger=logger,
                custom_callbacks=exp_custom_callbacks,
                lightning_model_kwargs=lightning_model_kwargs
            )
            logger.experiment.finish()
