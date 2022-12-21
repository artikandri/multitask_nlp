from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch.utils.data


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class used for a single task.

    Attributes:
        X (np.ndarray): Data with text and annotators ids.
        y (np.ndarray): Ground truth data.
        task_type (str): Task type.
        task_name (str, optional): Task name. Defaults to None.
        task_category (str, optional): Task category. Defaults to None.
        text_features (dict, optional): A dict mapping text feature keys with feature data. Defaults
            to None.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        task_name: Optional[str] = None,
        task_category: Optional[str] = None,
        text_features: Optional[Dict[str, Any]] = None
    ):
        self.X = X
        self.y = y
        self.task_type = task_type
        self.task_name = task_name
        self.task_category = task_category

        if text_features is not None:
            self.text_features = text_features
        else:
            self.text_features = {}

    def __getitem__(self, index) -> Tuple[Dict[str, Any], np.ndarray]:
        """Gets items from dataset for given indices.

        Args:
            index (int or list of int): Indices of requested data

        Returns:
            A tuple of (batch_data, batch_y) where batch_data is a dictionary which contains text
            ids, annotator ids, text features and information about task. batch_y is numpy array with
            ground truth data.
        """
        text_ids = self.X[index, 0]
        annotator_ids = self.X[index, 1]

        batch_data = {'text_ids': text_ids, 'annotator_ids': annotator_ids}
        for feature_key, text_feature in self.text_features.items():
            batch_data[feature_key] = text_feature[text_ids]

        batch_data['task_type'] = self.task_type
        batch_data['task_name'] = self.task_name
        batch_data['task_category'] = self.task_category

        batch_y = self.y[index]
        return batch_data, batch_y

    def __len__(self) -> int:
        """Returns number of elements in the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.y)
