import os
from copy import copy
from itertools import product
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from multitask_nlp.datasets.indonlu.emot_emotion_twitter import EmotEmotionTwitterDataModule
from multitask_nlp.learning.train_test import train_test, load_model, load_and_predict
from multitask_nlp.utils.file_loading import write_as_txt_file

from multitask_nlp.utils.analyze_models import get_params, get_size
from multitask_nlp.models import models as models_dict
from multitask_nlp.settings import CHECKPOINTS_DIR, LOGS_DIR
from multitask_nlp.utils import seed_everything

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_START_METHOD"] = "thread"

RANDOM_SEED = 2023
analyze_latest_model = True
ckpt_path = CHECKPOINTS_DIR / "bright-vortex-3"

if __name__ == "__main__":
    datamodule_cls = EmotEmotionTwitterDataModule

    rep_num = 1 if analyze_latest_model else 3
    model_types = ['multitask_transformer']
    model_names = ['labse']

    max_length = 256
    lr_rate = 1e-4
    epochs = 10
    batch_size = 32
    weight_decay = 0.01
    warmup_proportion = 0.1

    use_cuda = True
    custom_callbacks: List[pl.Callback] = [
        LearningRateMonitor()
    ]

    seed_everything()
    np.random.seed(RANDOM_SEED)

    data_module = datamodule_cls(batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup()

    for task_datamodule in [data_module]:
        dl = task_datamodule.train_dataloader()
        print(task_datamodule.task_name, end=' ')
        print(len(dl.dataset))


    lightning_model_kwargs = {
        'lr_scheduling': True,
        'warmup_proportion': warmup_proportion,
    }

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

            wandb_project_name = f'EmotEmotionTwitter_{model_name}_singleExp_EarlyStopping'
            
            if analyze_latest_model:
                if  os.path.exists(ckpt_path):
                    ckpt_files = os.listdir(ckpt_path)
                    if ckpt_files:
                        ckpt_file = ckpt_files[0]
                        model2 = load_model(model, ckpt_path=ckpt_path/ckpt_file)
                        size = get_size(model2)
                        total_params, trainable_params = get_params(model2)
                        exp_custom_callbacks = copy(custom_callbacks)
                        
                        predictions, avg_time = load_and_predict(
                            datamodule=data_module,
                            model=model2,
                            epochs=epochs,
                            lr=lr_rate,
                            logger=None,
                            exp_name=wandb_project_name,
                            weight_decay=weight_decay,
                            use_cuda=use_cuda,
                            custom_callbacks=exp_custom_callbacks,
                            lightning_model_kwargs=lightning_model_kwargs
                        )
                        
                        results = [wandb_project_name,
                                        f"ckpt_path: {ckpt_path}" ,
                                        f"model size: {size}" ,
                                        f"number of params: {total_params}",
                                        f"number of trainable params: {trainable_params}" ,
                                        f"average inference time: {avg_time}",
                                        f"nr of epochs: {epochs}",
                                        f"nr of rep: {i}"]
                                        
                        write_as_txt_file(results, f"{wandb_project_name}-{i}")  
                else:
                    print("checkpoint path doesnt exist")
            else:
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
                    lightning_model_kwargs=lightning_model_kwargs,
                )
                logger.experiment.finish()
