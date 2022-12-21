from torchmetrics import MeanAbsoluteError, MeanMetric, MeanSquaredError, R2Score

CLASSIFICATION_MEASURES = ('accuracy', 'precision', 'recall', 'f1', 'macro_f1')
REGRESSION_METRICS = {
    'mae': MeanAbsoluteError,
    'mse': MeanSquaredError,
    'r2': R2Score,
}

SEQUENCE_LABELLING_OVERALL_METRICS = {
    'macro_f1': MeanMetric,
    'micro_f1': MeanMetric,
    'accuracy': MeanMetric,
}

MACRO_AVG_MAE_TASKS = ['AllegroReviews']
POS_TAGGING_TASKS = ['CCPL', 'NKJP1M', 'PolEval18_POS']
KPWR_NER_HIERARCHICAL_TASKS = ['KPWr_n82_hierarchical']
