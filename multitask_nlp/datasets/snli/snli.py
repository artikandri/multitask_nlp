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


class SNLI_DataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = SNLI_DATA
        self.annotation_column = ['label']
        self.text_column = 'sentence1'
        self.text_2_column = 'sentence2'

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
