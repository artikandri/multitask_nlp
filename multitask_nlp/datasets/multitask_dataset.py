from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from multitask_nlp.datasets.base_dataset import BaseDataset


class MultiTaskDataset(torch.utils.data.Dataset, ABC):
    """Abstract class for multitask dataset.

    Attributes:
        task_dataloaders (list of DataLoader): Dataloaders for single tasks.
        task_iterators (list of iterator): Iterators for single task dataloaders.
    """

    def __init__(self, task_dataloaders: List[DataLoader], **kwargs):
        self.task_dataloaders = task_dataloaders
        self.task_iterators = [iter(dl) for dl in task_dataloaders]

    def __getitem__(self, index: int):
        """Gets items from multitask dataset for given indices.

        Return batch with data for some single task.

        Args:
            index (int)

        Returns:
            A tuple of (batch_data, batch_y) where batch_data is a dictionary which contains text
            ids, annotator ids, text features and information about task. batch_y is numpy array
            with ground truth data.
        """
        task_index = self._get_task_id(index)
        task_iterator = self.task_iterators[task_index]
        try:
            batch_data, batch_y = next(task_iterator)
        except StopIteration:
            print_msg = 'Iteration for task {}'.format(task_index)
            if isinstance(self.task_dataloaders[task_index].dataset, BaseDataset):
                print_msg += ': {}'.format(self.task_dataloaders[task_index].dataset.task_name)
            print_msg += ' ended'
            print(print_msg)

            task_iterator = iter(self.task_dataloaders[task_index])
            self.task_iterators[task_index] = task_iterator
            batch_data, batch_y = next(task_iterator)

        return batch_data, batch_y

    @abstractmethod
    def _get_task_id(self, index: int) -> int:
        """Returns task id for which batch will be yielded.

        It is an abstract method, each child class implements its own method for determining
        task id.

        Args:
            index (ind)

        Returns:
             int: task id
        """
        pass


class RoundRobinMTDataset(MultiTaskDataset):
    """Round-robin dataset. It ensures that in a single epoch all data for all tasks will be
    processed.

    That approach was for example used in "Multi-Task Deep Neural Networks for Natural Language
    Understanding" by Liu et al. (2019).

    Attributes:
        see base class for description of basic attributes.
        task_loaders_batches_indices (list of int): Task ids list. Number of ids in the list for one
            task is equal to number of batches so total length of the list is equals to the number
            of all batches coming from all single task dataloaders.
    """

    def __init__(self, task_dataloaders: List[DataLoader], **kwargs):
        super().__init__(task_dataloaders, **kwargs)

        task_loaders_batches_indices = []
        for i, dl in enumerate(task_dataloaders):
            task_loaders_batches_indices.extend([i] * len(dl))
        self.task_loaders_batches_indices = task_loaders_batches_indices

    def _get_task_id(self, index: int) -> int:
        return self.task_loaders_batches_indices[index]

    def __len__(self) -> int:
        return len(self.task_loaders_batches_indices)


class SequentialMTDataset(MultiTaskDataset):
    """Sequential dataset. It yields batches for subsequent tasks.

    Attributes:
        see base class for description of basic attributes.
        task_ids (list of int): Task ids list.
        steps (int): Number of steps (processed batches) in one epoch.
        current_index (ind): Index of task which batch will be returned in the next step.
    """

    def __init__(self, task_dataloaders: List[DataLoader], steps: int, **kwargs):
        super().__init__(task_dataloaders, **kwargs)
        self.task_ids = list(range(len(self.task_dataloaders)))
        self.steps = steps
        self.current_index = 0

    def _get_task_id(self, index: int) -> int:
        task_id = self.task_ids[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.task_ids)
        return task_id

    def __len__(self) -> int:
        return self.steps


class SamplingMTDataset(MultiTaskDataset):
    """Sampling dataset. It yields batches for randomly chosen task.

    Sampling is uniform for all tasks.

    Attributes:
        see base class for description of basic attributes.
        task_ids (list of int): Task ids list.
        steps (int): Number of steps (processed batches) in one epoch.
    """

    def __init__(self, task_dataloaders: List[DataLoader], steps: int, **kwargs):
        super().__init__(task_dataloaders, **kwargs)
        self.task_ids = list(range(len(self.task_dataloaders)))
        self.steps = steps

    def _get_task_id(self, index: int) -> int:
        return np.random.choice(self.task_ids)

    def __len__(self):
        return self.steps


class ProportionalSamplingMTDataset(SamplingMTDataset):
    """Proportional sampling dataset which yields batches for randomly chosen task.

    It inherits from SamplingMTDataset class. The proportional sampling was presented in
    "A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks" by Sanh et al.
    (2018).

    The probability of a given task is proportional to the single task dataset size so data from
    larger datasets will be yielded more frequently.

    Attributes:
        see base class for description of basic attributes.
    """

    def __init__(self, task_dataloaders: List[DataLoader], steps: int, alpha: float = 1.0,
                 **kwargs):
        """Creates proportional sampling dataset object.

        Args:
            task_dataloaders (list of DataLoader): Dataloaders for single tasks.
            steps (int): Number of steps (processed batches) in one epoch
            alpha (float, optional): Parameter controlling smoothing of probabilities. Default to
                1.0 which is no smoothing
        """
        super().__init__(task_dataloaders, steps, **kwargs)
        self._set_task_probs(alpha)

    def _get_task_id(self, index: int) -> int:
        return np.random.choice(self.task_ids, p=self.task_probs)

    def set_alpha(self, alpha: float) -> None:
        """Sets alpha parameter.

        Args:
            alpha (float)
        """
        self._set_task_probs(alpha)

    def _set_task_probs(self, alpha: float) -> None:
        """Sets probabilities of each task.

        Probabilities are proportional to their size with respect to smoothing parameter alpha. When
        alpha is equal 1.0 it is equivalent to standard non-smoothed probabilities proportional to
        task sizes. When it goes to 0.0 it becomes more and more smoothed (uniform).

        Args:
            alpha (float): Smoothing parameter.
        """
        task_probs = np.array([len(dl) ** alpha for dl in self.task_dataloaders])
        self.task_probs = task_probs / np.sum(task_probs)
