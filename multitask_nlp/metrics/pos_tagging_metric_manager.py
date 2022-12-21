from typing import Any, Dict, List, Optional, Set

from sklearn.metrics import accuracy_score
from torchmetrics import MeanMetric

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.metrics.metrics_settings import SEQUENCE_LABELLING_OVERALL_METRICS
from multitask_nlp.metrics.sequence_labelling_metric_manager import POS_TAGGING_TASKS, \
    SequenceLabellingMetricManager
from multitask_nlp.utils.nkjp_pos_tagging import ParsedTag
from multitask_nlp.utils.iob_tagging import from_iob_tag


class POS_TaggingMetricManager(SequenceLabellingMetricManager):
    """Metric manger for POS tagging tasks.

    It calculates standard sequence labelling metrics for each category of tags. Additionally, it
    aggregates all tag categories to calculate POS accuracy.
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
            if task in POS_TAGGING_TASKS:
                class_names = self.class_names[task]
                splits_considered_for_task = self.splits_to_consider[task]
                for split in splits_considered_for_task:
                    self._define_sequence_labelling_metrics_for_split(
                        class_names, metrics, split, task
                    )
                    # For POS tags we add additional accuracy metric which will measure accuracy
                    # of the reconstructed tag from its components
                    metrics[f'{task}_{split}_accuracy_pos'] = MeanMetric()

        return metrics

    def _get_handled_sequence_labelling_tasks(self, batch_outputs: List[Dict[str, Any]]) \
            -> Set[str]:
        seq_labeling_task_names = set()
        for batch_output in batch_outputs:
            task_name = batch_output['task_name']
            if task_name in POS_TAGGING_TASKS:
                seq_labeling_task_names.add(batch_output['task_name'])

        return seq_labeling_task_names

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
        """Calculates accuracy of POS tags.

        Tags in IOB format are transformed into standard format and parsed into full POS tags, which
        then are used to calculate exact accuracy.

        Args:
             See base class.
        """
        if task_name in POS_TAGGING_TASKS:
            total_tags = 0
            for cls_idx in range(len(class_dims)):
                class_name = class_names[cls_idx] if class_names else str(cls_idx)

                category_y_pred = [from_iob_tag(y_pred_label) for seq_y_pred in
                                   categories_y_pred[class_name] for y_pred_label in seq_y_pred]

                category_y_true = [from_iob_tag(y_true_label) for seq_y_true in
                                   categories_y_true[class_name] for y_true_label in seq_y_true]

                categories_y_true[class_name] = category_y_true
                categories_y_pred[class_name] = category_y_pred
                total_tags = len(category_y_true)

            full_tags_true = []
            full_tags_pred = []
            for i in range(total_tags):
                true_tag_object = ParsedTag(
                    **{k: v[i] for k, v in categories_y_true.items()}
                )
                full_tags_true.append(true_tag_object.to_full_tag())

                pred_tagset = {k: v[i] for k, v in categories_y_pred.items()}
                pred_tag_object = ParsedTag.create_consistent_tag(pred_tagset)
                full_tags_pred.append(pred_tag_object.to_full_tag())

            acc_score = accuracy_score(y_true=full_tags_pred, y_pred=full_tags_true)

            task_metric_key = f'{task_name}_{split}_accuracy_pos'
            self.metrics[task_metric_key](acc_score)
            metric_dict[task_metric_key] = self.metrics[task_metric_key]

            if log_metric_for_task_category:
                task_cat_metric_key = task_metric_key.replace(task_name, task_category)
                self.metrics[task_cat_metric_key](acc_score, len(task_batch_outputs))
                metric_dict[task_cat_metric_key] = self.metrics[task_cat_metric_key]
