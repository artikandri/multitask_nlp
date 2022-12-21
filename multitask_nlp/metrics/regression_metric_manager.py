from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch
from torchmetrics import MeanMetric

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.metrics.base_metric_manager import BaseMetricManager
from multitask_nlp.metrics.metrics_settings import MACRO_AVG_MAE_TASKS, REGRESSION_METRICS
from multitask_nlp.settings import MODEL_INPUT_TYPE, MODEL_OUTPUT_TYPE
from multitask_nlp.utils.metrics import macro_averaged_mean_absolute_error


class RegressionMetricManager(BaseMetricManager):
    """Metric manger for regression tasks.

    It manages regression metrics such as R2 or MSE.
    """

    def __init__(
        self,
        tasks_datamodules: List[BaseDataModule],
        extra_test_datamodules: Optional[List[BaseDataModule]] = None,
        tasks_to_not_log_detailed_metrics=None
    ):
        super().__init__(
            tasks_datamodules,
            extra_test_datamodules,
            tasks_to_not_log_detailed_metrics
        )

    def _define_metrics(self) -> Dict:
        metrics = {}
        for task in self.class_dims.keys():
            task_type = self.task_types[task]
            class_names = self.class_names[task]

            if task_type == 'regression':
                splits_considered_for_task = self.splits_to_consider[task]
                for split in splits_considered_for_task:
                    for metric_type, metric_class in REGRESSION_METRICS.items():
                        metric_objects = []
                        for class_idx in range(len(class_names)):
                            class_name = class_names[class_idx]
                            metric_obj = metric_class()
                            metrics[f'{task}_{split}_{metric_type}_{class_name}'] = metric_obj
                            metric_objects.append(metric_obj)

                        metrics[f'{task}_{split}_{metric_type}_mean'] = sum(metric_objects) / len(
                            metric_objects)

                    if task in MACRO_AVG_MAE_TASKS:
                        # Since we use imbalanced library to calculate weighted MAE score, we only
                        # aggregate it using MeanMetric class
                        metric_objects = []
                        for class_idx in range(len(class_names)):
                            class_name = class_names[class_idx]
                            metric_obj = MeanMetric()
                            metrics[f'{task}_{split}_w_mae_score_{class_name}'] = metric_obj
                            metric_objects.append(metric_obj)

                        mean_metric_obj = sum(metric_objects) / len(metric_objects)
                        metrics[f'{task}_{split}_w_mae_score_mean'] = mean_metric_obj

        return metrics

    def get_metrics_for_output(
        self,
        x: MODEL_INPUT_TYPE,
        output: MODEL_OUTPUT_TYPE,
        y: Union[torch.Tensor, np.ndarray],
        split: str
    ) -> Dict:
        """Returns regression metrics after each step.

        Args:
            See base class.

        Returns:
            See base class.
        """
        task_name = x['task_name']
        task_type = x['task_type']
        task_category = self.task_categories[task_name]

        log_detailed_metrics = task_name not in self.tasks_to_not_log_detailed_metrics
        # If task name is different from task category (see BaseDataModule to tell the difference)
        # then separate metric for task category is logged.
        log_metric_for_task_category = task_category != task_name

        class_names = self.class_names[task_name]

        metric_dict = {}
        if task_type in ['regression']:
            for metric_type in REGRESSION_METRICS.keys():
                for cls_idx, class_name in enumerate(class_names):
                    task_metric_key = f'{task_name}_{split}_{metric_type}_{class_name}'

                    metric_keys = [task_metric_key]
                    if log_metric_for_task_category:
                        metric_keys.append(task_metric_key.replace(task_name, task_category))

                    for metric_key in metric_keys:
                        # For R2 metric update method has to be used when batch is size of size
                        # less than 2, because standard __forward__ apart from updating internal
                        # state of metric object calculates metric values for current batch
                        # data (see https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
                        # for details) what is not possible for R2 measure when there is only one
                        # record.
                        if metric_type == 'r2' and len(output) < 2:
                            self.metrics[metric_key].update(output[:, cls_idx], y[:, cls_idx])
                        else:
                            self.metrics[metric_key](output[:, cls_idx], y[:, cls_idx])

                        if log_detailed_metrics:
                            metric_dict[metric_key] = self.metrics[metric_key]

                task_mean_metric_key = f'{task_name}_{split}_{metric_type}_mean'
                metric_keys = [task_mean_metric_key]
                if log_metric_for_task_category:
                    metric_keys.append(task_mean_metric_key.replace(task_name, task_category))

                for mean_metric_key in metric_keys:
                    metric_dict[mean_metric_key] = self.metrics[mean_metric_key]

        return metric_dict

    def get_metrics_at_epoch_end(self, split: str, batch_outputs: List[Dict[str, Any]]) -> Dict:
        """Returns regression metrics calculated at the end of epoch.

        For regression task, it is only used to calculated Macro average MAE metric which is not
        defined in Torchmetric package and needs data from the entire epoch to be calculated
        correctly

        Returns:
            See base class.
        """
        metric_dict = {}

        macro_avg_mae_tasks = set()
        for batch_output in batch_outputs:
            if batch_output['task_name'] in MACRO_AVG_MAE_TASKS:
                macro_avg_mae_tasks.add(batch_output['task_name'])

        for task_name in macro_avg_mae_tasks:
            task_category = self.task_categories[task_name]
            log_metric_for_task_category = task_category != task_name

            task_batch_outputs = [bo for bo in batch_outputs if bo['task_name'] == task_name]
            class_names = self.class_names[task_name]

            for cls_idx, class_name in enumerate(class_names):
                y_pred = None
                y_true = None

                for batch_output in task_batch_outputs:
                    b_y_pred = batch_output['y_pred'][:, cls_idx].cpu().numpy()
                    b_y_true = batch_output['y_true'][:, cls_idx].cpu().numpy()

                    y_pred = y_pred if y_pred is None else np.concatenate([y_pred, b_y_pred])
                    y_true = y_true if y_true is None else np.concatenate([y_true, b_y_true])

                w_mae = macro_averaged_mean_absolute_error(y_true, y_pred)
                w_mae_score = 1 - w_mae

                metric_key = f'{task_name}_{split}_w_mae_score_{class_name}'
                self.metrics[metric_key](w_mae_score)
                metric_dict[metric_key] = self.metrics[metric_key]

                if log_metric_for_task_category:
                    task_cat_metric_key = metric_key.replace(task_name, task_category)
                    self.metrics[task_cat_metric_key](w_mae_score, len(task_batch_outputs))
                    metric_dict[task_cat_metric_key] = self.metrics[task_cat_metric_key]

            mean_metric_key = f'{task_name}_{split}_w_mae_score_mean'
            metric_dict[mean_metric_key] = self.metrics[mean_metric_key]
            if log_metric_for_task_category:
                task_cat_mean_metric_key = mean_metric_key.replace(task_name, task_category)
                metric_dict[task_cat_mean_metric_key] = self.metrics[task_cat_mean_metric_key]

        return metric_dict
