from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import DYK_DATA

DEFAULT_RANDOM = 42


class DYK_DataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = DYK_DATA
        self.annotation_column = ['target']
        self.text_column = 'question'
        self.text_2_column = 'answer'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2]

    @property
    def task_name(self):
        return "DYK"

    @property
    def task_type(self):
        return 'classification'

    @property
    def texts_clean(self):
        return (self.data[self.text_column] + ' ' + self.data[self.text_2_column]).to_list()

    def prepare_data(self) -> None:
        text_df, annotations_df = self._get_data_from_split_files()
        self.data = text_df
        self.annotations = annotations_df

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        df_train_raw = pd.read_table(self.data_dir / 'train.tsv', delimiter=r"\t", engine='python')

        # We take as den and test sets train split, cause DYK is KLEJ benchmark dataset and the real test
        # split has no ground truth labels and in addition there is no dev set
        df_train, df_test = train_test_split(
            df_train_raw, test_size=1000,
            stratify=df_train_raw['target'],
            random_state=DEFAULT_RANDOM
        )
        df_dev, df_test = train_test_split(
            df_test, test_size=500,
            stratify=df_test['target'],
            random_state=DEFAULT_RANDOM
        )

        df_train['split'] = 'train'
        df_dev['split'] = 'dev'
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df = df.rename(columns={'q_id': 'text_id'})
        texts_df = df.loc[:, ['text_id', 'question', 'answer', 'split']]
        annotations_df = df.loc[:, ['text_id', 'target']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
