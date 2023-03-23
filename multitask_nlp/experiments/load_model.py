from typing import List, Optional, Union

import torch
import torch.nn as nn
from multitask_nlp.models import models as models_dict
from multitask_nlp.datasets import multitask_datasets as multitask_datasets_dict

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.datasets.multitask_datamodule import MultiTaskDataModule
from multitask_nlp.learning.lightning_model import Model
from multitask_nlp.datasets.indonlu.casa_absa_prosa import CasaAbsaProsaDataModule
from multitask_nlp.datasets.goemotions.goemotions import GoEmotionsDataModule
from multitask_nlp.datasets.studemo.studemo import StudEmoDataModule
from multitask_nlp.settings import CHECKPOINTS_DIR, LOGS_DIR
import pytorch_lightning as pl


model_cls = models_dict['multitask_transformer']
multitask_dataset_cls = multitask_datasets_dict["sampling"]

model_name = "indo-bert"
max_length = 512
batch_size = 8
epochs = 10
lr_rate = 1e-5
weight_decay = 0.01
lr_scheduling = True
warmup_proportion = 0.1
steps_in_epoch_list = [6500]
total_steps_list = [s * epochs for s in steps_in_epoch_list]
  
    
def get_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb
    
def load_model(
    model: nn.Module,
    ckpt_path=None,
):
        
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()

    return model

if __name__ == "__main__":
    
    
    task_datamodules_setup = {
        CasaAbsaProsaDataModule: {"batch_size": batch_size, "task_type": "classification"}, #emotions
        # StudEmoDataModule: {"batch_size": batch_size}, #emotions
    }
    task_to_not_log_detailed = []
    tasks_datamodules = []
    for datamodule_cls, datamodule_args in task_datamodules_setup.items():
        data_module = datamodule_cls(**datamodule_args)
        data_module.prepare_data()
        print(data_module)
        data_module.setup("predict")
        tasks_datamodules.append(data_module)
        
    model = model_cls(
        tasks_datamodules=tasks_datamodules,
        model_name=model_name,
        max_length=max_length,
    )
    multitask_dataset_args_list = []
    for total_steps in total_steps_list:
        epochs_steps = total_steps // epochs
        multitask_dataset_args_list.append({"steps": epochs_steps})
    mtl_datamodule = MultiTaskDataModule(
                        tasks_datamodules=tasks_datamodules,
                        multitask_dataset_cls=multitask_dataset_cls,
                        multitask_dataset_args=multitask_dataset_args_list
                    )
    path = CHECKPOINTS_DIR / "still-water-1/epoch=6-step=182.ckpt"
    model2 = load_model(datamodule=mtl_datamodule, model=model, ckpt_path=path)
    get_size(model2)
    pytorch_total_params = sum(p.numel() for p in model2.parameters())
    print("number of params", pytorch_total_params)
    pytorch_total_trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print("number of trainable params", pytorch_total_trainable_params)
    
    
    
    