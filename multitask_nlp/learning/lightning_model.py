from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.metrics.base_metric_manager import BaseMetricManager
from multitask_nlp.metrics.overall_score_metric_manager import OverallScoreMetricManager
from multitask_nlp.settings import GROUND_TRUTH_TYPE, MODEL_INPUT_TYPE, MODEL_OUTPUT_TYPE
from multitask_nlp.metrics import metric_manager_classes
from multitask_nlp.learning.loss import EqualWeightMultiTaskLoss, ScaledMultiTaskLoss, \
    ScalingType, UncertaintyWeightMultiTaskLoss


class Model(pl.LightningModule):
    """Lightning module class used in the project.

    It defines training, validation and test hooks along with optimizer and learning rate scheduler.
    It uses appropriate loss module during training and takes advantage of metrics manager
    objects which are used to calculate metrics which are then logged by Model.

    Attributes:
        model (torch.nn.Module): Torch module which forwards data and which parameters are updated
            during training.
        lr (float): Learning rate used by optimizer.
        weight_decay (float, optional): Weight decay parameter used for L2 regularization. Defaults
            to 0.0 so no regularization is applied.
        lr_scheduling (bool, optional): Whether to use linear lr scheduler or not. Default to False.
        warmup_proportion (float, optional): ratio of training steps during lr increases, only when
            lr_scheduling is True. Defaults to 0.1.
        uncertainty_loss_lr (float, optional): Learning rate value used to update log variances when
            uncertainty loss is used.
        task_class_dims (dict): A dict mapping tasks with their class dims.
        test_task_class_dims (dict): A dict mapping test tasks with their class dims.
        loss_module (torch.nn.Module): Torch module which forward function calculates loss.
        metric_managers (torch.nn.ModuleList): List of metric mangers.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks_datamodules: List[BaseDataModule],
        lr: float,
        weight_decay: float = 0.0,
        lr_scheduling: bool = False,
        warmup_proportion: float = 0.1,
        uncertainty_loss: bool = False,
        uncertainty_loss_lr: float = 2.5e-3,
        scaling_type: Optional[ScalingType] = None,
        tasks_to_not_log_detailed_metrics: List[str] = None,
        extra_test_datamodules: Optional[List[BaseDataModule]] = [],
        **kwargs
    ):
        """Initializes Lightning Module.

        Args:
            tasks_datamodules (list of BaseDataModule): Tasks for which train, val and test metrics
                will be calculated
            uncertainty_loss (bool, optional): Whether to use uncertainty loss or not. Defaults to
                False.
            scaling_type (ScalingType, optional): Scaling type used for ScaledMultiTaskLoss. When it
                is not None ScaledMultiTaskLoss is used. In that case, it should be either
                ScalingType.LINEAR or ScalingType.LOG. Defaults to None. If uncertainty_loss is
                True, ScaledMultiTaskLoss is not used regardless of scaling_type value.
            tasks_to_not_log_detailed_metrics (list of str): Task for which only basic metrics
                should be logged.
            extra_test_datamodules (list of BaseDataModule): Tasks for which only test metrics
                will be calculated.
            **kwargs (): Extra keyword arguments.
            See the class documentation for description of the remaining parameters.
        """
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduling = lr_scheduling
        self.warmup_proportion = warmup_proportion
        self.uncertainty_loss_lr = uncertainty_loss_lr

        self.task_class_dims = {
            task_dm.task_name: task_dm.class_dims for task_dm in tasks_datamodules
        }
        self.test_task_class_dims = {
            task_dm.task_name: task_dm.class_dims for task_dm in extra_test_datamodules
        }
        all_task_class_dims = {** self.task_class_dims, ** self.test_task_class_dims}

        if uncertainty_loss:
            self.loss_module = UncertaintyWeightMultiTaskLoss(all_task_class_dims)
        elif scaling_type is not None:
            self.loss_module = ScaledMultiTaskLoss(all_task_class_dims, scaling_type)
        else:
            self.loss_module = EqualWeightMultiTaskLoss(all_task_class_dims)

        tasks_to_not_log_detailed_metrics = tasks_to_not_log_detailed_metrics or []

        metric_managers: List[BaseMetricManager] = []
        for metric_manager_class in metric_manager_classes:
            metric_manager_obj = metric_manager_class(
                tasks_datamodules=tasks_datamodules,
                extra_test_datamodules=extra_test_datamodules,
                tasks_to_not_log_detailed_metrics=tasks_to_not_log_detailed_metrics,
            )
            metric_managers.append(metric_manager_obj)

        overall_metric_manager = OverallScoreMetricManager(
            metric_managers=metric_managers,
            tasks_datamodules=tasks_datamodules,
            extra_test_datamodules=extra_test_datamodules,
            tasks_to_not_log_detailed_metrics=tasks_to_not_log_detailed_metrics
        )
        metric_managers.append(overall_metric_manager)
        self.metric_managers = nn.ModuleList(metric_managers)
        self.save_hyperparameters()
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def forward(self, x: MODEL_INPUT_TYPE):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        if self.lr_scheduling:
            lr_scheduler = self._get_linear_lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step"
                }
            }
        return optimizer

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Creates AdamW optimizer used to update model parameters.

        Appropriate regularization and learning rate parameters are used.

        Returns:
            Optimizer used during training stage.
        """
        n_param_optimizer = list(self.named_parameters())
        variance_params = ['loss_module.log_variances']
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in n_param_optimizer if not any(nd in n for nd in no_decay) and
                        not any(ep in n for ep in variance_params)],
             'weight_decay': self.weight_decay,
             'name': 'weight_decay_pg'},
            {'params': [p for n, p in n_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'name': 'non_weight_decay_pg'},
        ]
        specific_variance_params = [p for n, p in n_param_optimizer if
                                    any(ep in n for ep in variance_params)]
        if len(specific_variance_params) > 0:
            optimizer_grouped_parameters.append(
                {'params': specific_variance_params, 'weight_decay': 0.0,
                 'lr': self.uncertainty_loss_lr, 'name': 'variance_params'}
            )

        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

    def _get_linear_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> \
            torch.optim.lr_scheduler.LambdaLR:
        """Configures linear learning rate scheduler.

        It calculates number of steps during training which is needed to determine how long lr is
        increased and then decreased.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer which learning rate will be scheduled.

        Returns:
            Learning rate scheduler.
        """
        self.trainer.reset_train_dataloader(self)
        train_loader = self.trainer.train_dataloader

        n_batch = len(train_loader)
        n_accumulate_grad = self.trainer.accumulate_grad_batches
        n_max_epochs = self.trainer.max_epochs
        n_devices = self.trainer.num_gpus or 1

        num_training_steps = (n_batch // n_accumulate_grad) * n_max_epochs // n_devices
        num_warmup_steps = int(self.warmup_proportion * num_training_steps)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return lr_scheduler

    def _shared_step(self, batch: Tuple[MODEL_INPUT_TYPE, GROUND_TRUTH_TYPE]) \
            -> Tuple[str, str, MODEL_OUTPUT_TYPE, torch.Tensor]:
        """Performs forward pass of the model.

        Args:
            batch (tuple): A tuple of X data and ground truth data.

        Returns:
            A tuple of (task_name, task_type, output, loss) where output comes from the model, and
            loss is a tensor with calculated loss for the model output against the ground truth
            using loss_module.
        """
        x, y_true = batch
        output = self.forward(x)
        task_type = x['task_type']
        task_name = x['task_name']

        loss = self.loss_module.forward(x, y_true, output)
        return task_name, task_type, output, loss


    def training_step(self, batch, batch_idx, optimizer_idx=None):
        task_name, task_type, output, loss = self._shared_step(batch)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self._log_losses_per_tasks(task_name, loss, 'train')

        if isinstance(self.loss_module, UncertaintyWeightMultiTaskLoss):
            for task_name, log_variance in self.loss_module.log_variances.items():
                variance = torch.exp(log_variance)
                self.log(f"{task_name}_variance", variance.item(),
                         on_step=True, on_epoch=False, prog_bar=False)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        task_name, task_type, output, loss = self._shared_step(batch)
        self.log("valid_loss", loss.item(), on_epoch=True, prog_bar=True)
        self._log_losses_per_tasks(task_name, loss, 'valid')

        x, y_true = batch
        self._log_metrics_at_step_end(x, output, y_true, "valid")
        return {"valid_loss": loss, 'y_pred': output, 'y_true': y_true,
                'task_name': task_name, 'task_type': task_type}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        task_name, task_type, output, loss = self._shared_step(batch)
        self.log("test_loss", loss.item(), on_epoch=True, prog_bar=True)
        self._log_losses_per_tasks(task_name, loss, 'test')

        x, y_true = batch
        self._log_metrics_at_step_end(x, output, y_true, "test")
        return {"test_loss": loss, 'y_pred': output, 'y_true': y_true,
                'task_name': task_name, 'task_type': task_type}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        task_name, task_type, output, loss = self._shared_step(batch)
        self.log("predict_loss", loss.item(), on_epoch=True, prog_bar=True)
        self._log_losses_per_tasks(task_name, loss, 'predict')
        torch.cuda.empty_cache()
        minibatch_model_in, _ = batch
        self.starter.record()
        _ = self(minibatch_model_in)
        self.ender.record()
        torch.cuda.synchronize()
        inference_time = self.starter.elapsed_time(self.ender)
        print("time", inference_time*1e-3)
                
        x, y_true = batch
        self._log_metrics_at_step_end(x, output, y_true, "predict")
        return {"predict_loss": loss, 'y_pred': output, 'y_true': y_true,
                'task_name': task_name, 'task_type': task_type}
            

    def validation_epoch_end(self, outputs):
        self._log_metrics_at_epoch_end(split='valid', batch_outputs=outputs)

    def test_epoch_end(self, outputs) -> None:
        # When many test dataloaders are passed in Trainer test function, outputs are in the form
        # of the list containing step outputs for separate dataloaders which has to be flattened.
        if isinstance(outputs[0], list):
            outputs = [o for dataloader_outputs in outputs for o in dataloader_outputs]

        self._log_metrics_at_epoch_end(split='test', batch_outputs=outputs)

    def _log_losses_per_tasks(self, current_task_name: str, loss: torch.Tensor, split: str) -> None:
        """Logs losses for particular tasks.

        It logs loss for currently processed task. For the other tasks logged loss is set to 0.

        Args:
            current_task_name (str): Current task name. The task which is being processed.
            loss (torch.Tensor): Loss value.
            split (str): Dataset split.
        """
        tasks_to_log_loss = list(self.task_class_dims.keys())
        if split == 'test':
            tasks_to_log_loss += list(self.test_task_class_dims.keys())

        for task_name in tasks_to_log_loss:
            if task_name != current_task_name:
                self.log(f"{task_name}_{split}_loss", 0.0, on_step=True, on_epoch=True,
                         reduce_fx=torch.sum)

        self.log(f"{current_task_name}_{split}_loss", loss.item(), on_step=True,
                 on_epoch=True, reduce_fx=torch.sum)

    def _log_metrics_at_step_end(
        self,
        x: MODEL_INPUT_TYPE,
        output: MODEL_OUTPUT_TYPE,
        y: Union[torch.Tensor, np.ndarray],
        split: str
    ) -> None:
        """Logs metrics at the step end.

        The metrics are calculated based on the current step model output and the corresponding
        ground truth. All available metrics managers are used.

        Args:
            x: Model input
            output: Model output.
            y: Ground truth data..
            split (str): Dataset split.
        """
        log_dict = {}
        for metric_manager in self.metric_managers:
            log_dict.update(metric_manager.get_metrics_for_output(x, output, y, split))

        self.log_dict(log_dict)

    def _log_metrics_at_epoch_end(self, split: str, batch_outputs: List[Dict[str, Any]]) -> None:
        """Logs metrics at the epoch end.

        The metrics are calculated based on all batches processed during epoch.
        All available metrics managers are used.

        Args:
            split (str): Dataset split.
            batch_outputs (list of dict): List containing batch data and their model outputs.
        """
        for metric_manager in self.metric_managers:
            # try/except used because general score not always can be calculated and
            # then an ValueError is raised.
            try:
                log_dict = metric_manager.get_metrics_at_epoch_end(split, batch_outputs)
                self.log_dict(log_dict)
            except ValueError:
                pass
