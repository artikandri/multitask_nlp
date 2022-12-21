import math
from enum import Enum
from typing import Dict, List, Union

import numpy as np
import torch
from torch import nn

from multitask_nlp.settings import GROUND_TRUTH_TYPE, MODEL_INPUT_TYPE, MODEL_OUTPUT_TYPE


class BaseMultiTaskLoss(nn.Module):
    """Base MultiTask loss module class.

    It defines forward function which calculates loss with respect to the task type.

    Attributes:
          task_class_dims (dict): A dict matching tasks with their class dimensionalities.
    """
    def __init__(
        self,
        task_class_dims: Dict[str, List[int]],
    ):
        super().__init__()
        self.task_class_dims = task_class_dims

    def forward(
        self,
        x: MODEL_INPUT_TYPE,
        y_true: GROUND_TRUTH_TYPE,
        output: MODEL_OUTPUT_TYPE
    ) -> torch.Tensor:
        """Calculates loss for given model input and output.

        Args:
            x: Model input.
            y_true: Ground truth data.
            output: Model output.

        Returns:
            torch.Tensor: Loss value.

        Raises:
            ValueError: If task_type is not correct.
        """
        task_type = x['task_type']
        task_name = x['task_name']

        if task_type in ['classification', 'sequence labeling']:
            loss = 0
            class_dims = self.task_class_dims[task_name]

            if task_type == 'sequence labeling':
                # In the case of sequence labelling, ground truth labels per each token in each
                # text have to be represented as a tensor.
                y_true_tensors = []
                for seq_y_true in y_true:
                    seq_y_true = np.vstack(seq_y_true).T
                    seq_y_true = torch.tensor(seq_y_true, dtype=torch.long)
                    y_true_tensors.append(seq_y_true)

                y_true = y_true_tensors

            for cls_idx in range(len(class_dims)):
                start_idx = sum(class_dims[:cls_idx])
                end_idx = start_idx + class_dims[cls_idx]

                if task_type == 'classification':
                    loss = loss + nn.CrossEntropyLoss()(
                        output[:, start_idx:end_idx],
                        y_true[:, cls_idx].long()
                    )
                else:  # sequence labeling case
                    # In the case of sequence labelling, predicted labels per each token in each
                    # text are concatenated.
                    cls_output_cat = torch.cat([seq_y_pred[:, start_idx:end_idx]
                                                for seq_y_pred in output])

                    # Ground truth labels per each token for each text are concatenated. There are
                    # only considered those tags which are predicted by the model (it is possible
                    # that model do not predict tag for every token because of too low max_length).
                    cls_y_true = []
                    for seq_y_true, seq_y_pred in zip(y_true, output):
                        seq_y_true = seq_y_true[:len(seq_y_pred), cls_idx].long()
                        cls_y_true.append(seq_y_true)

                    cls_y_true_cat = torch.cat(cls_y_true).to(cls_output_cat.device)

                    loss = loss + nn.CrossEntropyLoss()(cls_output_cat, cls_y_true_cat)

        elif task_type == 'regression':
            y_true = y_true.float()
            loss = nn.MSELoss()(output, y_true)
        else:
            raise ValueError

        return loss


class EqualWeightMultiTaskLoss(BaseMultiTaskLoss):
    """Equal Weight MultiTask loss module class.

    Loss calculated in the base class is divided by the number of all tasks.

    Attributes:
          See the base class.
    """
    def __init__(self, task_class_dims: Dict[str, List[int]]):
        super().__init__(task_class_dims)

    def forward(
        self,
        x: MODEL_INPUT_TYPE,
        y_true: Union[torch.Tensor, np.ndarray],
        output: MODEL_OUTPUT_TYPE
    ) -> torch.Tensor:
        loss = super().forward(x, y_true, output)
        return loss / len(self.task_class_dims)


class ScalingType(Enum):
    LINEAR = 'linear'
    LOG = 'log'


class ScaledMultiTaskLoss(BaseMultiTaskLoss):
    """Scaled MultiTask loss module class.

    The method used in "Muppet: Massive Multi-task Representations with Pre-Finetuning" by
    Aghajanyan (2021).

    In the case of classification or sequence labelling, loss calculated in the base class is
    divided by the number of possible classes to predict (in the linear scaling), e.g., for binary
    classification it will be divided by 2, for classification from 4 categories it will be
    divided by 4.

    For task where many categories are to be predicted, the loss is divided by sum of possible
    classes from each category.

    Attributes:
          scaling_type (ScalingType): Dictates scaling type. Either linear or log.
    """
    def __init__(self, task_class_dims: Dict[str, List[int]], scaling_type: ScalingType):
        super().__init__(task_class_dims)
        self.scaling_type = scaling_type

    def forward(
        self,
        x: MODEL_INPUT_TYPE,
        y_true: Union[torch.Tensor, np.ndarray],
        output: MODEL_OUTPUT_TYPE
    ) -> torch.Tensor:

        loss = super().forward(x, y_true, output)

        task_type = x['task_type']
        task_name = x['task_name']

        if task_type in ['classification', 'sequence labeling']:
            class_dims = self.task_class_dims[task_name]
            if self.scaling_type == ScalingType.LINEAR:
                loss = loss / sum(class_dims)
            elif self.scaling_type == ScalingType.LOG:
                loss = loss / math.log(sum(class_dims))
            else:
                raise ValueError('scaling_type must be either linear or log.')

        return loss / len(self.task_class_dims)


class UncertaintyWeightMultiTaskLoss(BaseMultiTaskLoss):
    """Uncertainty Weight MultiTask loss module class.

    The loss proposed in "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
    and Semantics" by Kendall (2018).

    For each task, its loss is scaled by some variance value which is changed throughout the
    training, so the importance of each task can be dynamically adjusted.

    Attributes:
        log_variances (torch.nn.ParameterDict): A dictionary mapping each task with its log
            variance learnable parameter.
    """
    def __init__(self, task_class_dims: Dict[str, List[int]]):
        super().__init__(task_class_dims)
        self.log_variances = nn.ParameterDict(
            {
                task: nn.Parameter(data=torch.tensor([0.0], requires_grad=True))
                for task in self.task_class_dims.keys()
            }
        )

    def forward(
        self,
        x: MODEL_INPUT_TYPE,
        y_true: Union[torch.Tensor, np.ndarray],
        output: MODEL_OUTPUT_TYPE
    ) -> torch.Tensor:
        loss = super().forward(x, y_true, output)

        log_variance = self.log_variances[x['task_name']]
        inverted_variance = torch.exp(-log_variance)
        loss = inverted_variance * loss + log_variance
        return loss
