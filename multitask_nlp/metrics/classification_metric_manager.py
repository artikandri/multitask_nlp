from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.metrics.base_metric_manager import BaseMetricManager
from multitask_nlp.metrics.metrics_settings import CLASSIFICATION_MEASURES
from multitask_nlp.settings import MODEL_INPUT_TYPE, MODEL_OUTPUT_TYPE


class ClassificationMetricManager(BaseMetricManager):
    """Metric manager for classification tasks.

    It manages classification metrics such as accuracy, F1 score etc.
    """
    def __init__(
        self,
        tasks_datamodules: List[BaseDataModule],
        extra_test_datamodules: Optional[List[BaseDataModule]] = None,
        tasks_to_not_log_detailed_metrics=None
    ):
        super().__init__(tasks_datamodules, extra_test_datamodules,
                         tasks_to_not_log_detailed_metrics)

    def _define_metrics(self) -> Dict:
        metrics = {}
        for task in self.class_dims.keys():
            task_type = self.task_types[task]
            class_dims = self.class_dims[task]
            class_names = self.class_names[task]

            if task_type == 'classification':
                splits_considered_for_task = self.splits_to_consider[task]
                for split in splits_considered_for_task:

                    macro_f1_metrics = []
                    for class_idx in range(len(class_dims)):
                        class_name = class_names[class_idx] if class_names else str(class_idx)
                        num_classes = class_dims[class_idx]

                        metrics[f'{task}_{split}_accuracy_{class_name}'] = Accuracy()
                        metrics[f'{task}_{split}_precision_{class_name}'] = Precision(
                            num_classes=num_classes, average='none')
                        metrics[f'{task}_{split}_recall_{class_name}'] = Recall(
                            num_classes=num_classes, average='none')
                        metrics[f'{task}_{split}_f1_{class_name}'] = F1Score(
                            num_classes=num_classes, average='none')

                        macro_f1_metric = F1Score(average='macro', num_classes=num_classes)
                        metrics[f'{task}_{split}_macro_f1_{class_name}'] = macro_f1_metric
                        macro_f1_metrics.append(macro_f1_metric)

                    metrics[f'{task}_{split}_macro_f1_mean'] = sum(macro_f1_metrics) / len(
                        macro_f1_metrics)

        return metrics

    def get_metrics_for_output(
        self,
        x: MODEL_INPUT_TYPE,
        output: MODEL_OUTPUT_TYPE,
        y: Union[torch.Tensor, np.ndarray],
        split: str
    ) -> Dict:
        """Returns classification metrics after each step.

        Returned are those metrics which are single value metric or are averaged for all classes.

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
        metric_for_task_category = task_category != task_name

        class_dims = self.class_dims[task_name]
        class_names = self.class_names[task_name]

        metric_dict = {}
        if task_type in ['classification']:
            output = torch.softmax(output, dim=1)

            for cls_idx in range(len(class_dims)):
                start_idx = sum(class_dims[:cls_idx])
                end_idx = start_idx + class_dims[cls_idx]

                class_name = class_names[cls_idx] if class_names else str(cls_idx)
                for metric_type in CLASSIFICATION_MEASURES:
                    task_metric_key = f'{task_name}_{split}_{metric_type}_{class_name}'

                    metric_keys = [task_metric_key]
                    if metric_for_task_category:
                        metric_keys.append(task_metric_key.replace(task_name, task_category))

                    for metric_key in metric_keys:
                        metric_value = self.metrics[metric_key](
                            output[:, start_idx:end_idx].float(),
                            y[:, cls_idx].int()
                        )
                        # On step stage, only metrics with a single value are logged, e.g.,
                        # accuracy, macro F1. Other classification metrics are calculated per each
                        # class separately, thus they cannot be logged as torchmetric object.
                        if log_detailed_metrics and not metric_value.size():
                            metric_dict[metric_key] = self.metrics[metric_key]

            for metric_type in CLASSIFICATION_MEASURES:
                task_mean_metric_key = f'{task_name}_{split}_{metric_type}_mean'
                metric_keys = [task_mean_metric_key]
                if metric_for_task_category:
                    metric_keys.append(task_mean_metric_key.replace(task_name, task_category))

                for mean_metric_key in metric_keys:
                    if mean_metric_key in self.metrics:
                        metric_dict[mean_metric_key] = self.metrics[mean_metric_key]

        return metric_dict

    def get_metrics_at_epoch_end(self, split: str, batch_outputs: List[Dict[str, Any]]) -> Dict:
        """Returns classification metrics which are calculated individually per each class.

        In other words, returned are those metrics which are not averaged between classes.

        Returns:
            See base class.
        """
        metric_dict = {}
        for metric_key, metric in self.metrics.items():
            if split in metric_key:
                metric = self.metrics[metric_key]
                if hasattr(metric, 'average') and metric.average in [None, 'none']:
                    metric_value = self.metrics[metric_key].compute()

                    log_metric = True
                    for task_not_to_log in self.tasks_to_not_log_detailed_metrics:
                        if task_not_to_log in metric_key:
                            log_metric = False
                            break

                    if log_metric:
                        for idx in range(metric_value.size(dim=0)):
                            metric_dict[f'{metric_key}_{idx}'] = metric_value[idx]

                    self.metrics[metric_key].reset()

        return metric_dict
