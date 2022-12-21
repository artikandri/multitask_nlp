from typing import Any, Dict, List, Optional, Set

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.metrics.metrics_settings import KPWR_NER_HIERARCHICAL_TASKS, \
    SEQUENCE_LABELLING_OVERALL_METRICS
from multitask_nlp.metrics.sequence_labelling_metric_manager import SequenceLabellingMetricManager
from multitask_nlp.utils.kpwr_ner_tagging import KPWr_NER_ParsedTag


class KPWr_NER_MetricManager(SequenceLabellingMetricManager):
    """Metric manger for KPWr task in hierarchical form.

    It calculates seqeval metrics but using tags aggregated from all levels (categories).
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
            if task in KPWR_NER_HIERARCHICAL_TASKS:
                splits_considered_for_task = self.splits_to_consider[task]
                for split in splits_considered_for_task:
                    # Since we use seqeval library to calculate sequence labeling methods, we only
                    # aggregate accuracy and F1 (micro/macro) using MeanMetric class.
                    # Metrics are reported for different NER depth levels.
                    for metric_type, metric_class in SEQUENCE_LABELLING_OVERALL_METRICS.items():
                        for level_i in [1, 2, 3]:
                            class_name = 'ner'
                            if level_i < 3:
                                class_name = f'ner_level_{level_i}'
                            metrics[f'{task}_{split}_{metric_type}_{class_name}'] = metric_class()

        return metrics

    def _get_handled_sequence_labelling_tasks(self, batch_outputs: List[Dict[str, Any]]) -> Set[str]:
        seq_labeling_task_names = set()
        for batch_output in batch_outputs:
            task_name = batch_output['task_name']
            if task_name in KPWR_NER_HIERARCHICAL_TASKS:
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
        """Calculates sequence labelling for NER tags aggregated from level tag categories.

        Args:
             See base class.
        """
        if task_name in KPWR_NER_HIERARCHICAL_TASKS:
            parsed_true_tags = []
            parsed_pred_tags = []

            for text_i in range(len(categories_y_true['category'])):
                text_true_tags = [KPWr_NER_ParsedTag(
                    **{k: v[text_i][tag_i] for k, v in categories_y_true.items()}
                ) for tag_i in range(len(categories_y_true['category'][text_i]))]
                parsed_true_tags.append(text_true_tags)

                text_pred_tags = [KPWr_NER_ParsedTag.create_consistent_tag(
                    {k: v[text_i][tag_i] for k, v in categories_y_pred.items()}
                ) for tag_i in range(len(categories_y_pred['category'][text_i]))]
                parsed_pred_tags.append(text_pred_tags)

            for level_i in [1, 2, 3]:
                true_tags = [[parsed_tag.to_full_tag(level_i) for parsed_tag in text_tags]
                             for text_tags in parsed_true_tags]
                pred_tags = [[parsed_tag.to_full_tag(level_i) for parsed_tag in text_tags]
                             for text_tags in parsed_pred_tags]

                class_name = 'ner'
                if level_i < 3:
                    class_name = f'ner_level_{level_i}'

                self._get_metrics_for_sequecne_labelling(
                    y_pred=pred_tags, y_true=true_tags, split=split, task_name=task_name,
                    metric_dict=metric_dict, task_category=task_category, class_name=class_name,
                    task_batch_outputs=task_batch_outputs,
                    log_metric_for_task_category=log_metric_for_task_category
                )
