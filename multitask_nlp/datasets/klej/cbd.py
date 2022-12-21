from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import CBD_DATA

DEFAULT_RANDOM = 42


class CBD_DataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = CBD_DATA
        self.annotation_column = ['target']
        self.text_column = 'text'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2]

    @property
    def task_name(self):
        return "CBD"

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
        df_train_raw = pd.read_table(self.data_dir / 'train.tsv', delimiter=r"\t", engine='python')
        df_train_raw = df_train_raw.dropna(subset=['target'])

        # We take as den and test sets train split, cause CBD is KLEJ benchmark dataset and the real test
        # split has no ground truth labels and in addition there is no dev set
        df_train, df_test = train_test_split(
            df_train_raw, test_size=2000,
            stratify=df_train_raw['target'],
            random_state=DEFAULT_RANDOM
        )
        df_dev, df_test = train_test_split(
            df_test, test_size=1000,
            stratify=df_test['target'],
            random_state=DEFAULT_RANDOM
        )

        df_train = df_train.reset_index(drop=True)
        df_train['text_id'] = 'train_' + df_train.index.astype(str)
        df_train['split'] = 'train'

        df_dev = df_dev.reset_index(drop=True)
        df_dev['text_id'] = 'dev_' + df_dev.index.astype(str)
        df_dev['split'] = 'dev'

        df_test = df_test.reset_index(drop=True)
        df_test['text_id'] = 'dev_' + df_test.index.astype(str)
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df = df.rename(columns={'sentence': 'text'})
        texts_df = df.loc[:, ['text_id', 'text', 'split']]
        annotations_df = df.loc[:, ['text_id', 'target']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
