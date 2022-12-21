from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

import torch

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.metrics.base_metric_manager import BaseMetricManager
from multitask_nlp.metrics.metrics_settings import KPWR_NER_HIERARCHICAL_TASKS, MACRO_AVG_MAE_TASKS, \
    POS_TAGGING_TASKS
from multitask_nlp.settings import MODEL_INPUT_TYPE, MODEL_OUTPUT_TYPE


class OverallScoreMetricManager(BaseMetricManager):
    """Metric manger for overall score over all tasks.

    It takes main metrics for each task and define overall score as the sum of them. The overall
    score is calculated at the end of epoch taking main metrics from appropriate metric managers.

    Attributes:
        metric_managers (list of BaseMetricManager): List of used metric mangers.
    """

    def __init__(
        self,
        metric_managers: Sequence[BaseMetricManager],
        tasks_datamodules: List[BaseDataModule],
        extra_test_datamodules: Optional[List[BaseDataModule]] = None,
        tasks_to_not_log_detailed_metrics=None,
    ):
        self.metric_managers = metric_managers
        super().__init__(
            tasks_datamodules, extra_test_datamodules, tasks_to_not_log_detailed_metrics
        )

    def _define_metrics(self) -> Dict:
        metrics = {}

        task_main_measures = {}
        for task_name in self.task_types.keys():
            task_main_measure_name = self._get_main_measure_name_for_task(task_name)
            task_main_measures[task_name] = task_main_measure_name

        for split in ['valid', 'test']:
            task_main_metrics = []
            for task_name, measure in task_main_measures.items():
                metric_key = f'{task_name}_{split}_{measure}'

                for metric_manager in self.metric_managers:
                    if metric_key in metric_manager.metrics:
                        metric = metric_manager.metrics[f'{task_name}_{split}_{measure}']
                        task_main_metrics.append(metric)
                        break

            if len(task_main_metrics) > 0:
                metrics[f'{split}_overall_score'] = sum(task_main_metrics) / len(task_main_metrics)

        return metrics

    def _get_main_measure_name_for_task(self, task_name: str) -> str:
        """Returns name of main metric for given task.

        Args:
            task_name (str): Task name.

        Returns:
            str: Main measure name for task.
        """
        measure_type = self._get_main_measure_type_for_task(task_name)
        class_names = self.class_names[task_name]

        if task_name in POS_TAGGING_TASKS:
            task_main_measure_name = f'{measure_type}'
        elif task_name in KPWR_NER_HIERARCHICAL_TASKS:
            task_main_measure_name = f'{measure_type}_ner'
        elif len(class_names) == 1:
            class_name = class_names[0]
            task_main_measure_name = f'{measure_type}_{class_name}'
        else:
            task_main_measure_name = f'{measure_type}_mean'
        return task_main_measure_name

    def _get_main_measure_type_for_task(self, task_name: str) -> str:
        """Returns type of main metric for given task.

        Args:
            task_name (str): Task name.

        Returns:
            str: Main measure type for task.

        Raises:
            ValueError: When task type is incorrect.
        """
        task_type = self.task_types[task_name]
        if task_type == 'classification':
            measure_type = 'macro_f1'
        elif task_type == 'sequence labeling':
            if task_name in POS_TAGGING_TASKS:
                measure_type = 'accuracy_pos'
            else:
                measure_type = 'micro_f1'
        elif task_type == 'regression':
            if task_name in MACRO_AVG_MAE_TASKS:
                measure_type = 'w_mae_score'
            else:
                measure_type = 'r2'
        else:
            raise ValueError(f"Error, {task_type} is incorrect task type.")

        return measure_type

    def get_metrics_for_output(
        self,
        x: MODEL_INPUT_TYPE,
        output: MODEL_OUTPUT_TYPE,
        y: Union[torch.Tensor, np.ndarray],
        split: str
    ) -> Dict:
        """Returns empty dictionary."""
        return {}

    def get_metrics_at_epoch_end(self, split: str, batch_outputs: List[Dict[str, Any]]) -> Dict:
        overall_metric_key = f'{split}_overall_score'
        self.metrics[overall_metric_key].compute()
        return {overall_metric_key: self.metrics[overall_metric_key]}
