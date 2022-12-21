from typing import List

import numpy as np
import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import MEASURING_HATE_SPEECH_DATA

DEFAULT_SPLITS = [0.6, 0.2, 0.2]


class MeasuringHateSpeechDataModule(BaseDataModule):
    def __init__(
        self,
        normalize: bool = False,
        split_sizes: List[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if split_sizes is None:
            split_sizes = DEFAULT_SPLITS
        assert len(split_sizes) == 3 and sum(split_sizes) == 1 and all(0 <= s <= 1 for s in split_sizes)

        self.data_dir = MEASURING_HATE_SPEECH_DATA
        self.split_sizes = split_sizes
        self.annotation_column = ['hate_speech_score']
        self.text_column = 'text'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

        self.normalize = normalize

    @property
    def class_dims(self):
        return [1]

    @property
    def task_name(self):
        return "MeasuringHateSpeech"

    @property
    def task_type(self):
        return 'regression'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.data_dir / 'data.csv')
        self.annotations = pd.read_csv(self.data_dir / 'annotations.csv').dropna()

        if self.normalize:
            self.normalize_labels()

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
