from enum import IntEnum
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import CASA_ABSA_PROSA_DATA

ANNOTATION_COLUMNS = [
    'fuel','machine','others','part','price','service'
]

class Labels(IntEnum):
    positive = 0
    neutral = 1
    negative = 2


_CLASS_MAPPING = {
    "positive": Labels.positive,
    "neutral": Labels.neutral,
    "negative": Labels.negative,
}


class CasaAbsaProsaDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = CASA_ABSA_PROSA_DATA
        self.annotation_column = ANNOTATION_COLUMNS
        self.text_column = 'text'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [3] * 6

    @property
    def task_name(self):
        return "casa_absa-prosa"

    @property
    def task_type(self):
        return 'classification'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_df, annotations_df = self._get_data_from_split_files()
        self.data = text_df
        self.annotations = annotations_df

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        df_train = self._load_df(self.data_dir / 'train_preprocess.csv')
        df_dev = self._load_df(self.data_dir / 'valid_preprocess.csv')
        df_test = self._load_df(self.data_dir / 'test_preprocess.csv')

        df_train['split'] = 'train'
        df_dev['split'] = 'dev'
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df['text_id'] = df.index

        texts_df = df.loc[:, ['text_id', 'text', 'split']]
        annotations_df = self._get_annotations(df)
        return texts_df, annotations_df

    def _get_annotations(self, df):
        for column in ANNOTATION_COLUMNS:
            df[column].replace(_CLASS_MAPPING, inplace=True)
        df.loc[:, 'annotator_id'] = 0
        return df

    @staticmethod
    def _load_df(directory: Path) -> pd.DataFrame:
        df = pd.read_csv(directory)
        df['text'] = df['sentence'] 
        return df
