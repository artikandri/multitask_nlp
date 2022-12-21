import os
from itertools import product
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

from multitask_nlp.datasets.nkjp1m.nkjp1m import NKJP1M_DataModule
from multitask_nlp.learning.train_test import train_test
from multitask_nlp.models import models as models_dict
from multitask_nlp.settings import LOGS_DIR
from multitask_nlp.utils import seed_everything

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

RANDOM_SEED = 42

if __name__ == "__main__":
    datamodule_cls = NKJP1M_DataModule
    rep_num = 3

    model_types = ['multitask_transformer']
    model_names = ['polish-distilroberta', 'polish-roberta']
    wandb_project_name = 'NKJP1MPOS_EXP'

    max_length = 256
    lr_rate = 1e-5
    epochs = 4
    batch_size = 32
    weight_decay = 0.1
    warmup_proportion = 0.06

    use_cuda = True
    custom_callbacks: List[pl.Callback] = [
        LearningRateMonitor()
    ]
    lightning_model_kwargs = {
        'lr_scheduling': True,
        'warmup_proportion': warmup_proportion
    }

    seed_everything()

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

            train_test(
                datamodule=data_module,
                model=model,
                epochs=epochs,
                lr=lr_rate,
                weight_decay=weight_decay,
                use_cuda=use_cuda,
                logger=logger,
                custom_callbacks=custom_callbacks,
                lightning_model_kwargs=lightning_model_kwargs
            )
            logger.experiment.finish()