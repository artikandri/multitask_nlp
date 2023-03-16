import os
from copy import copy
from itertools import product
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from multitask_nlp.datasets import multitask_datasets as multitask_datasets_dict
from multitask_nlp.datasets.indonlu.casa_absa_prosa import CasaAbsaProsaDataModule
from multitask_nlp.datasets.indonlu.facqa_qa_factoid_itb import FacqaQaFactoidItbDataModule
from multitask_nlp.datasets.indonlu.wrete_entailment_ui import WreteEntailmentUiDataModule
from multitask_nlp.datasets.goemotions.goemotions import GoEmotionsDataModule
from multitask_nlp.datasets.studemo.studemo import StudEmoDataModule
from multitask_nlp.datasets.snli.snli import SNLI_DataModule
from multitask_nlp.datasets.multitask_datamodule import MultiTaskDataModule
from multitask_nlp.datasets.indonesian_emotion.indonesian_emotion import IndonesianEmotionDataModule
from multitask_nlp.datasets.indonlu.nerp_ner_prosa import NerpNerProsaDataModule
from multitask_nlp.datasets.indonlu.smsa_doc_sentiment_prosa import SmsaDocSentimentProsaDataModule
from multitask_nlp.datasets.conll2003.conll2003 import Conll2003DataModule

from multitask_nlp.learning.train_test import train_test
from multitask_nlp.models import models as models_dict
from multitask_nlp.settings import CHECKPOINTS_DIR, LOGS_DIR
from multitask_nlp.utils import seed_everything
from multitask_nlp.utils.callbacks.dynamic_proportion_sampling import AnnealingSampling, \
    DynamicTemperatureSampling
from multitask_nlp.utils.callbacks.mtl_dataloader_manager import ValidDatasetResetter

os.environ["WANDB_START_METHOD"] = "thread"

use_cuda = True
RANDOM_SEED = 2023

stl_experiments = False


def run_experiments():
    model_types = ['multitask_transformer']
    model_names = ['xlmr', 'indo-roberta']
    rep_num = 5

    loss_args_list = [(False, None)]
    multitask_dataset_types = ['sampling']

    max_length = 512
    batch_size = 16
    epochs = 10
    lr_rate = 1e-5
    weight_decay = 0.01
    lr_scheduling = True
    warmup_proportion = 0.1

    trainer_kwargs = {
        'accumulate_grad_batches': 2
    }
    custom_callbacks: List[pl.Callback] = [
        ValidDatasetResetter()
    ]

    steps_in_epoch_list = [5500]
    total_steps_list = [s * epochs for s in steps_in_epoch_list]

    # proportional sampling arguments
    alpha_list = [0.2]

    # dynamic temperature arguments
    N_list = [3]
    T0_list = [1]
    T_max_list = [5]

    task_datamodules_setup = {
        #GoEmotionsDataModule: {"batch_size": batch_size}, #emotions
        # StudEmoDataModule: {"batch_size": batch_size}, #emotions
        IndonesianEmotionDataModule: {"batch_size": batch_size}, #emotions
        CasaAbsaProsaDataModule: {"batch_size": batch_size}, #sentiment
        SmsaDocSentimentProsaDataModule: {"batch_size": batch_size}, #sentiment analysis
        # WreteEntailmentUiDataModule: {"batch_size": batch_size}, #entailment
        # SNLI_DataModule:  {"batch_size": batch_size}, #entailment
        # FacqaQaFactoidItbDataModule: {"batch_size": batch_size}, #ner
        # NerpNerProsaDataModule: {"batch_size": batch_size}, #ner 
        # Conll2003DataModule: {"batch_size": batch_size}, #ner
    }
    task_to_not_log_detailed = ['GoEmotions']

    lightning_model_kwargs = {
        'lr_scheduling': lr_scheduling,
        'warmup_proportion': warmup_proportion,
        'tasks_to_not_log_detailed_metrics': task_to_not_log_detailed
    }

    tasks_datamodules = []
    for datamodule_cls, datamodule_args in task_datamodules_setup.items():
        data_module = datamodule_cls(**datamodule_args)
        data_module.prepare_data()
        data_module.setup()
        tasks_datamodules.append(data_module)

    total_train = 0
    for tasks_datamodule in tasks_datamodules:
        dl = tasks_datamodule.train_dataloader()
        print(tasks_datamodule.task_name, end=' ')
        print(len(dl.dataset))
        total_train += len(dl.dataset)

    print('Total size of all:', total_train)
    print('Steps in RR epoch:', total_train // batch_size)

    seed_everything()
    for model_type, model_name, loss_args in \
        product(model_types, model_names, loss_args_list):
        model_cls = models_dict[model_type]

        wandb_project_name = f'MTL_mix2_id_{model_name}_emocls_EarlyStopping'

        uncertainty_loss, scaling_type = loss_args
        hparams = {
            "model_type": model_type,
            "model_name": model_name,
            "num_epochs": epochs,
            "learning_rate": lr_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "warmup_proportion": warmup_proportion,
            "max_length": max_length,
            "uncertainty_loss": uncertainty_loss,
            "scaling_type": scaling_type
        }

        used_lightning_model_kwargs = copy(lightning_model_kwargs).update({
            'uncertainty_loss': uncertainty_loss,
            'scaling_type': scaling_type
        })

        for i in range(rep_num):
            # # Single-Task learning
            if stl_experiments:
                for data_module in tasks_datamodules:
                    torch.manual_seed(RANDOM_SEED + i)
                    model = model_cls(
                        tasks_datamodules=[data_module],
                        model_name=model_name,
                        max_length=max_length,
                    )
                    hparams_copy = copy(hparams)
                    hparams_copy["learning_kind"] = 'STL'
                    hparams_copy["dataset"] = data_module.task_name
                    run_training(
                        model, data_module, hparams_copy, epochs, lr_rate, weight_decay,
                        custom_callbacks=custom_callbacks,
                        lightning_model_kwargs=used_lightning_model_kwargs,
                        trainer_kwargs=trainer_kwargs
                    )

            # Multi-Task learning
            for multitask_dataset_type in multitask_dataset_types:
                multitask_dataset_cls = multitask_datasets_dict[multitask_dataset_type]

                mtl_extra_callbacks = []
                if multitask_dataset_type == 'proportional_sampling':
                    multitask_dataset_args_list = []
                    for total_steps, alpha in product(total_steps_list, alpha_list):
                        epochs_steps = total_steps // epochs
                        multitask_dataset_args_list.append({"steps": epochs_steps, "alpha": alpha})

                elif multitask_dataset_type == 'sampling':
                    multitask_dataset_args_list = []
                    for total_steps in total_steps_list:
                        epochs_steps = total_steps // epochs
                        multitask_dataset_args_list.append({"steps": epochs_steps})

                elif multitask_dataset_type == 'annealing_sampling':
                    multitask_dataset_args_list = []
                    for total_steps in total_steps_list:
                        epochs_steps = total_steps // epochs
                        multitask_dataset_args_list.append({"steps": epochs_steps})

                    mtl_extra_callbacks.append(AnnealingSampling(epochs_number=epochs))

                elif multitask_dataset_type == 'dynamic_temperature_sampling':
                    multitask_dataset_args_list = []
                    for total_steps in total_steps_list:
                        epochs_steps = total_steps // epochs
                        multitask_dataset_args_list.append({"steps": epochs_steps})

                    for N, T_0, T_max in product(N_list, T0_list, T_max_list):
                        dynamic_temp_sampling_callback = DynamicTemperatureSampling(
                            N=N, T_0=T_0, T_max=T_max
                        )
                        mtl_extra_callbacks.append(dynamic_temp_sampling_callback)
                else:
                    multitask_dataset_args_list = [{}]

                if len(mtl_extra_callbacks) == 0:
                    mtl_extra_callbacks.append(None)

                for multitask_dataset_args, mtl_extra_callback in \
                    product(multitask_dataset_args_list, mtl_extra_callbacks):

                    mtl_custom_callbacks = copy(custom_callbacks)
                    if mtl_extra_callback is not None:
                        mtl_custom_callbacks.append(mtl_extra_callback)

                    mtl_datamodule = MultiTaskDataModule(
                        tasks_datamodules=tasks_datamodules,
                        multitask_dataset_cls=multitask_dataset_cls,
                        multitask_dataset_args=multitask_dataset_args
                    )
                    torch.manual_seed(RANDOM_SEED + i)
                    model = model_cls(
                        tasks_datamodules=tasks_datamodules,
                        model_name=model_name,
                        max_length=max_length,
                    )
                    hparams_copy = copy(hparams)
                    hparams_copy["learning_kind"] = 'MTL'
                    datasets_string = '_'.join([dm.task_name for dm in tasks_datamodules])
                    hparams_copy["dataset"] = datasets_string
                    hparams_copy["mt_dataset_type"] = multitask_dataset_type
                    hparams_copy.update(multitask_dataset_args)

                    run_training(
                        model, mtl_datamodule, hparams_copy, epochs, lr_rate, weight_decay,
                        custom_callbacks=mtl_custom_callbacks,
                        lightning_model_kwargs=used_lightning_model_kwargs,
                        trainer_kwargs=trainer_kwargs, 
                        project_name=wandb_project_name
                    )



def run_training(model, datamodule, hparams, epochs, lr_rate, weight_decay, custom_callbacks,
                 lightning_model_kwargs=None, trainer_kwargs=None, project_name="mtl"):

    logger = pl_loggers.WandbLogger(
        save_dir=str(LOGS_DIR),
        config=hparams,
        project=project_name,
        log_model=False,
    )

    run_custom_callbacks = copy(custom_callbacks)
    run_custom_callbacks.extend(
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
        datamodule=datamodule,
        model=model,
        epochs=epochs,
        lr=lr_rate,
        weight_decay=weight_decay,
        use_cuda=use_cuda,
        logger=logger,
        custom_callbacks=run_custom_callbacks,
        lightning_model_kwargs=lightning_model_kwargs,
        trainer_kwargs=trainer_kwargs
    )
    logger.experiment.finish()


if __name__ == "__main__":
    run_experiments()
