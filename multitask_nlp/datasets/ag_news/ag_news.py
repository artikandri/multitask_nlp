from enum import IntEnum
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import AG_NEWS_DATA

DEFAULT_RANDOM = 42


class Labels(IntEnum):
    WORLDS = 0
    SPORT = 1
    BUSINESS = 2
    SCITECH = 3


_CLASS_MAPPING = {
    1: Labels.WORLDS,
    2: Labels.SPORT,
    3: Labels.BUSINESS,
    4: Labels.SCITECH
}


class AG_NewsDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = AG_NEWS_DATA
        self.annotation_column = ['category']
        self.text_column = 'text'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [4]

    @property
    def task_name(self):
        return "AG News"

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
        df_train_raw = self._load_df(self.data_dir / 'train.csv')
        df_train, df_dev = train_test_split(
            df_train_raw, test_size=7600,
            random_state=DEFAULT_RANDOM,
            stratify=df_train_raw['category']
        )
        df_test = self._load_df(self.data_dir / 'test.csv')

        df_train['split'] = 'train'
        df_dev['split'] = 'dev'
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df['text_id'] = df.index

        texts_df = df.loc[:, ['text_id', 'text', 'split']]
        annotations_df = df.loc[:, ['text_id', 'category']]
        annotations_df.loc[:, 'annotator_id'] = 0
        annotations_df['category'].replace(_CLASS_MAPPING, inplace=True)
        return texts_df, annotations_df

    @staticmethod
    def _load_df(directory: Path) -> pd.DataFrame:
        df = pd.read_csv(directory)
        df['text'] = df['Title'] + ' ' + df['Description']
        df = df.rename(columns={'Class Index': 'category'})
        df = df.loc[:, ['text', 'category']]
        return df
