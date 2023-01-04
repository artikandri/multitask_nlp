from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import math

import torch
from seqeval.metrics import accuracy_score as seq_eval_accuracy_score, classification_report

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.metrics.base_metric_manager import BaseMetricManager
from multitask_nlp.metrics.metrics_settings import KPWR_NER_HIERARCHICAL_TASKS, POS_TAGGING_TASKS, \
    SEQUENCE_LABELLING_OVERALL_METRICS
from multitask_nlp.settings import MODEL_INPUT_TYPE, MODEL_OUTPUT_TYPE


class SequenceLabellingMetricManager(BaseMetricManager):
    """Metric manger for sequence labelling task.

    It uses seqeval library to calculate metrics. It allows calculating metrics for IOB format.

    This class is also a parent class for specific sequence labeling metric managers which use
    nested or hierarchical sequence labelling. Hence, processing of outputs need to be performed. It
    is done in _log_metrics_based_on_all_categories method which has to be overridden.
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

            # Class handles only those sequence labeling task which are nor KPWr hierarchical tasks
            # neither POS tagging tasks. Those task have specific metric manager classes.
            if task_type == 'sequence labeling' and \
                task not in KPWR_NER_HIERARCHICAL_TASKS + POS_TAGGING_TASKS:
                class_names = self.class_names[task]
                splits_considered_for_task = self.splits_to_consider[task]
                for split in splits_considered_for_task:
                    self._define_sequence_labelling_metrics_for_split(
                        class_names, metrics, split, task
                    )

        return metrics

    @staticmethod
    def _define_sequence_labelling_metrics_for_split(
        class_names: List[str],
        metrics: Dict[str, Any],
        split: str,
        task: str
    ) -> None:
        # Since we use seqeval library to calculate sequence labeling methods, we only
        # aggregate accuracy and F1 (micro/macro) using MeanMetric class (defined in
        # SEQUENCE_LABELLING_OVERALL_METRICS).
        for metric_type, metric_class in SEQUENCE_LABELLING_OVERALL_METRICS.items():
            metric_objects = []
            for class_idx in range(len(class_names)):
                class_name = class_names[class_idx]
                metric_obj = metric_class()
                metrics[f'{task}_{split}_{metric_type}_{class_name}'] = metric_obj
                metric_objects.append(metric_obj)

            metrics[f'{task}_{split}_{metric_type}_mean'] = \
                sum(metric_objects) / len(metric_objects)

    def get_metrics_for_output(
        self,
        x: MODEL_INPUT_TYPE,
        output: MODEL_OUTPUT_TYPE,
        y: Union[torch.Tensor, np.ndarray],
        split: str
    ) -> Dict:
        """Returns empty dictionary.

        Seqeval needs all epoch data to calculate metrics, thus on step epochs empty dictionary is
        returned.
        """
        return {}

    def get_metrics_at_epoch_end(self, split: str, batch_outputs: List[Dict[str, Any]]) -> Dict:
        """Returns sequence labelling metrics which are calculated using seqeval library.

         Returns:
            See base class.
        """

        metric_dict = {}
        seq_labelling_task_names = self._get_handled_sequence_labelling_tasks(batch_outputs)

        for task_name in seq_labelling_task_names:
            task_category = self.task_categories[task_name]
            log_metric_for_task_category = task_category != task_name

            task_batch_outputs = [bo for bo in batch_outputs if bo['task_name'] == task_name]

            categories_y_pred = {}
            categories_y_true = {}

            class_dims = self.class_dims[task_name]
            class_names = self.class_names[task_name]
            label_maps = self.label_maps[task_name]

            for cls_idx in range(len(class_dims)):
                start_idx = sum(class_dims[:cls_idx])
                end_idx = start_idx + class_dims[cls_idx]
                class_name = class_names[cls_idx] if class_names else str(cls_idx)
                label_map = label_maps[cls_idx]
                rev_label_map = {v: k for k, v in label_map.items()}

                y_pred = []
                y_true = []
                for batch_output in task_batch_outputs:
                    b_y_pred = [torch.argmax(seq_y_pred[:, start_idx:end_idx], dim=1).tolist()
                                for seq_y_pred in batch_output['y_pred']]

                    b_y_true = []
                    for seq_y_true, seq_y_pred in zip(batch_output['y_true'], b_y_pred):
                        seq_y_true = np.vstack(seq_y_true).T
                        seq_y_true = seq_y_true[:len(seq_y_pred), cls_idx].tolist()
                        b_y_true.append(seq_y_true)

                    y_pred.extend(b_y_pred)
                    y_true.extend(b_y_true)

                y_pred = [list(map(lambda t: rev_label_map[t], (0 if math.isnan(tag_seq) else tag_seq))) for tag_seq in y_pred]
                y_true = [list(map(lambda t: rev_label_map[t], (0 if math.isnan(tag_seq) else tag_seq))) for tag_seq in y_true]

                categories_y_true[class_name] = y_true
                categories_y_pred[class_name] = y_pred

                if task_name not in KPWR_NER_HIERARCHICAL_TASKS:
                    self._get_metrics_for_sequecne_labelling(
                        y_pred, y_true, split, task_name, class_name, metric_dict,
                        task_category, task_batch_outputs, log_metric_for_task_category
                    )

            if task_name not in KPWR_NER_HIERARCHICAL_TASKS:
                for metric_type in ['accuracy', 'macro_f1', 'micro_f1']:
                    task_mean_metric_key = f'{task_name}_{split}_{metric_type}_mean'
                    metric_keys = [task_mean_metric_key]
                    if log_metric_for_task_category:
                        task_cat_metric_key = task_mean_metric_key.replace(task_name, task_category)
                        metric_keys.append(task_cat_metric_key)

                    for mean_metric_key in metric_keys:
                        metric_dict[mean_metric_key] = self.metrics[mean_metric_key]

            self._log_metrics_based_on_all_categories(
                categories_y_pred, categories_y_true,
                split, task_name, class_names, class_dims,
                metric_dict,
                task_category, task_batch_outputs,
                log_metric_for_task_category
            )

        return metric_dict

    def _get_handled_sequence_labelling_tasks(self, batch_outputs: List[Dict[str, Any]]) -> Set[str]:
        """Returns set of task names for which metrics will be calculated.

        It can be overridden by child class to handle other (specific sequence labelling) tasks.

        Args:
             batch_outputs (list of dict): List containing batch data and their model outputs.

        Returns:
            (list of str); List of tasks.
        """
        seq_labeling_task_names = set()
        for batch_output in batch_outputs:
            if batch_output['task_type'] == 'sequence labeling':
                task_name = batch_output['task_name']
                if task_name not in KPWR_NER_HIERARCHICAL_TASKS + POS_TAGGING_TASKS:
                    seq_labeling_task_names.add(batch_output['task_name'])

        return seq_labeling_task_names

    def _get_metrics_for_sequecne_labelling(
        self,
        y_pred: List[List[str]],
        y_true: List[List[str]],
        split: str,
        task_name: str,
        class_name: str,
        metric_dict: dict,
        task_category: str,
        task_batch_outputs: List[Dict[str, Any]],
        log_metric_for_task_category: bool
    ) -> None:
        """Adds to metric dictionary, metric values obtained using seqeval library.

        Args:
            y_pred (list of list of str): Each list element consist predicted tags in IOB format for
             a single text.
            y_true (list of list of str): Each list element consist true tags in IOB format for
             a single text.
            split (str): Dataset split.
            task_name (str): Task name.
            class_name (str): Name of class (category) for which metric is predicted.
            metric_dict (dict): Dict of metrics which will be appended with new metrics.
            task_category (str): Task category.
            task_batch_outputs (list of dict): List containing batch data and their model outputs
                for current task.
            log_metric_for_task_category (bool): Whether to log separate metrics for task
                category.
        """
        log_detailed_metrics = task_name not in self.tasks_to_not_log_detailed_metrics

        report = classification_report(y_true, y_pred, output_dict=True)
        for category, metrics in report.items():
            if category not in ['micro avg', 'macro avg', 'weighted avg']:
                for metric_name, metric_val in metrics.items():
                    if metric_name not in ['support'] and log_detailed_metrics:
                        metric_key = f'{task_name}_{split}_{metric_name}_{class_name}_{category}'
                        metric_dict[metric_key] = metric_val

        acc_val = seq_eval_accuracy_score(y_true, y_pred)
        macro_f1_val = report['macro avg']['f1-score']
        micro_f1_val = report['micro avg']['f1-score']

        for metric_value, metric_type in zip(
            [acc_val, macro_f1_val, micro_f1_val],
            ['accuracy', 'macro_f1', 'micro_f1']
        ):
            task_metric_key = f'{task_name}_{split}_{metric_type}_{class_name}'
            self.metrics[task_metric_key](metric_value)
            metric_dict[task_metric_key] = self.metrics[task_metric_key]

            if log_metric_for_task_category:
                task_category_metric_key = task_metric_key.replace(task_name, task_category)
                self.metrics[task_category_metric_key](metric_value, len(task_batch_outputs))
                metric_dict[task_category_metric_key] = self.metrics[task_category_metric_key]

    def _log_metrics_based_on_all_categories(
        self,
        categories_y_pred: Dict[str, List[List[str]]],
        categories_y_true: Dict[str, List[List[str]]],
        split: str,
        task_name: str,
        class_names: List[str],
        class_dims: List[int],
        metric_dict: dict,
        task_category: str,
        task_batch_outputs: List[Dict[str, Any]],
        log_metric_for_task_category: bool
    ) -> None:
        """Adds to metric dictionary, metric values calculated using predictions and truth value
        using all tag categories.

        Args:
            categories_y_pred (dict): A dictionary which matches a tag category name with list which
                elements contain predicted tags in IOB format.
            categories_y_true (list of list of str):  A dictionary which matches a tag category name
                with list which elements contain true tags in IOB format.
            split (str): Dataset split.
            task_name (str): Task name.
            class_names (list of str): Names of class (tag categories).
            class_dims (list of int): Dimensions of task.
            metric_dict (dict): Dict of metrics which will be appended with new metrics.
            task_category (str): Task category.
            task_batch_outputs (list of dict): List containing batch data and their model outputs
                for current task.
            log_metric_for_task_category (bool): Whether to log separate metrics for task
                category.
        """
        pass
