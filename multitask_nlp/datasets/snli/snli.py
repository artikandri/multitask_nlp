from enum import IntEnum
from typing import List

import pandas as pd
import numpy as np

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import SNLI_DATA

DEFAULT_RANDOM = 42


class Labels(IntEnum):
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2


_CLASS_MAPPING = {
    "contradiction": Labels.CONTRADICTION,
    "neutral": Labels.NEUTRAL,
    "entailment": Labels.ENTAILMENT
}

DEFAULT_SPLITS = [0.6, 0.2, 0.2]


class SNLI_DataModule(BaseDataModule):
    def __init__(
        self,
        split_sizes: List[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = SNLI_DATA
        self.annotation_column = ['label']
        self.text_column = 'sentence1'
        self.text_2_column = 'sentence2'

        if split_sizes is None:
            split_sizes = DEFAULT_SPLITS
        assert len(split_sizes) == 3 and sum(split_sizes) == 1 and all(
            0 <= s <= 1 for s in split_sizes)


        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [3]

    @property
    def task_name(self):
        return "SNLI"

    @property
    def task_type(self):
        return 'classification'

    @property
    def texts_clean(self):
        return (self.data[self.text_column] + ' ' + self.data[self.text_2_column]).to_list()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / 'text_data.csv')
        self.annotations = pd.read_csv(self.data_dir / 'annotations.csv').dropna()
        self.annotations['label'].replace(_CLASS_MAPPING, inplace=True)
        self._assign_splits()

    def _assign_splits(self):
        data = self.data

        train_ratio, valid_ratio, test_ration = self.split_sizes
        train_idx = int(train_ratio * len(data))
        valid_idx = int(valid_ratio * len(data)) + train_idx

        indexes = np.arange(len(data.index))
        np.random.shuffle(indexes)

        data['split'] = ''
        data.iloc[indexes[:train_idx], data.columns.get_loc('split')] = 'train'
        data.iloc[indexes[train_idx:valid_idx], data.columns.get_loc('split')] = 'dev'
        data.iloc[indexes[valid_idx:], data.columns.get_loc('split')] = 'test'

        self.data = data