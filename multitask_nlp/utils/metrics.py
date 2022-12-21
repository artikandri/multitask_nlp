import numpy as np
from sklearn.metrics import mean_absolute_error


def macro_averaged_mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate macro average MAE.

    Macro average MAE is average of MAE calculated per each class (possible ground truth value).

    Args:
        y_true (np.ndaaray): Ground truth values.
        y_pred (np.ndaaray): Predicted values.

    Returns:
        (float) macro average MAE.
    """
    labels = np.sort(np.unique(y_true))
    mae = []
    for possible_class in labels:
        indices = np.flatnonzero(y_true == possible_class)

        mae.append(
            mean_absolute_error(
                y_true[indices],
                y_pred[indices]
            )
        )

    return np.sum(mae) / len(mae)
