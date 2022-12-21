from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch
import torch.nn as nn

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import MODEL_INPUT_TYPE, MODEL_OUTPUT_TYPE


class BaseMetricManager(nn.Module, ABC):
    """Base abstract class for metric manager.

    Attributes:
        tasks_to_not_log_detailed_metrics (list of str, optional): Tasks for which only main metrics
            are logged, for example due to too massive number of classes and categories
        class_dims (dict): A dict matching task names with their dimensionalities.
        class_names (dict): A dict matching task names with names of their categories (classes)
            to predict.
        task_types (dict): A dict matching task names with their task types.
        task_categories (dict): A dict matching task names with their task categories.
        label_maps (dict): A dict matching task names with their label maps which map encoded labels
            actual labels. Only for sequence labelling task for which actual labels are needed to
            calculate metrics.
        splits_to_consider (dict): A dict matching task names with dataset splits for which metrics
            should be calculated. It is usually "train", "val" and "test" but for
            extra_test_datamodules only "test" split is taken into account.
        metrics (torch.nn.ModuleDict): A dictionary matching metric names with their
            Torchmetric objects.
    """

    def __init__(
        self,
        tasks_datamodules: List[BaseDataModule],
        extra_test_datamodules: Optional[List[BaseDataModule]] = None,
        tasks_to_not_log_detailed_metrics: Optional[List[str]] = None
    ):
        super().__init__()
        self.tasks_to_not_log_detailed_metrics = tasks_to_not_log_detailed_metrics or []

        if extra_test_datamodules is None:
            extra_test_datamodules = []

        all_tasks_datamodules = tasks_datamodules + extra_test_datamodules

        self.class_dims = {task_dm.task_name: task_dm.class_dims for task_dm in
                           all_tasks_datamodules}
        self.class_names = {task_dm.task_name: task_dm.annotation_column for task_dm in
                            all_tasks_datamodules}
        self.task_types = {task_dm.task_name: task_dm.task_type for task_dm in
                           all_tasks_datamodules}
        self.task_categories = {task_dm.task_name: task_dm.task_category
                                for task_dm in all_tasks_datamodules}
        self.label_maps = {
            task_dm.task_name: task_dm.label_maps for task_dm in all_tasks_datamodules if
            task_dm.task_type == 'sequence labeling'
        }
        self.splits_to_consider = {task_dm.task_name: ['train', 'valid', 'test']
                                   for task_dm in tasks_datamodules}
        self.splits_to_consider.update(
            {task_dm.task_name: ['test'] for task_dm in extra_test_datamodules}
        )

        metrics = self._define_metrics()

        # for tasks which task category is different from its name, additional task categories
        # metrics are added. They can be common for many tasks with the same category.
        for task_name, task_category in self.task_categories.items():
            if task_name != task_category:
                if task_name in self.tasks_to_not_log_detailed_metrics:
                    self.tasks_to_not_log_detailed_metrics.append(task_category)

                for key in [k for k in metrics.keys() if f'{task_name}_' in k]:
                    task_category_metric_key = key.replace(task_name, task_category)
                    if task_category_metric_key not in metrics:
                        metric_obj = metrics[key]
                        metrics[task_category_metric_key] = copy(metric_obj)

        self.metrics = nn.ModuleDict(metrics)

    @abstractmethod
    def _define_metrics(self) -> Dict:
        """Defines and returns dictionary with metric objects."""
        pass

    @abstractmethod
    def get_metrics_for_output(
        self,
        x: MODEL_INPUT_TYPE,
        output: MODEL_OUTPUT_TYPE,
        y: Union[torch.Tensor, np.ndarray],
        split: str
    ) -> Dict:
        """Calculates metrics for given model step inputs and outputs.

        It is an abstract method. Depending on metric manager different task types and metrics will
        be handled.

        Args:
            x: Model input.
            output: Model output.
            y: Ground truth data.
            split (str): Dataset split.

        Returns:
            A dictionary with mapping metric names with their metric object or values.
        """
        pass

    @abstractmethod
    def get_metrics_at_epoch_end(self, split: str, batch_outputs: List[Dict[str, Any]]) -> Dict:
        """Calculates metrics for outputs from entire epoch.

        It is an abstract method. Depending on metric manager other task types and metrics will
        be considered.

        Args:
            split (str): Dataset split.
            batch_outputs (list of dict): List containing batch data and their model outputs.

        Returns:
            A dictionary with mapping metric names with their metric object or values.
        """
        pass
