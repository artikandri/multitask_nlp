from multitask_nlp.metrics.classification_metric_manager import ClassificationMetricManager
from multitask_nlp.metrics.kpwr_ner_hierarchial_metric_manager import KPWr_NER_MetricManager
from multitask_nlp.metrics.pos_tagging_metric_manager import POS_TaggingMetricManager
from multitask_nlp.metrics.regression_metric_manager import RegressionMetricManager
from multitask_nlp.metrics.sequence_labelling_metric_manager import SequenceLabellingMetricManager

metric_manager_classes = (
    ClassificationMetricManager,
    RegressionMetricManager,
    SequenceLabellingMetricManager,
    POS_TaggingMetricManager,
    KPWr_NER_MetricManager
)
