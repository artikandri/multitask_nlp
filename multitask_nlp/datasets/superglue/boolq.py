from enum import IntEnum
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import BOOLQ_DATA

DEFAULT_RANDOM = 42


class Labels(IntEnum):
    FALSE = 0
    TRUE = 1


_CLASS_MAPPING = {
    False: Labels.FALSE,
    True: Labels.TRUE
}


class BoolQDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = BOOLQ_DATA
        self.annotation_column = ['label']
        self.text_column = 'question'
        self.text_2_column = 'passage'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2]

    @property
    def task_name(self):
        return "BoolQ"

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
        self.annotations['label'].replace(_CLASS_MAPPING, inplace=True)

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        df_train_raw = pd.read_json(self.data_dir / 'train.jsonl', lines=True)

        df_train, df_dev = train_test_split(df_train_raw, test_size=2500,
                                            random_state=DEFAULT_RANDOM)
        df_train['idx'] = 'train_' + df_train['idx'].astype(str)
        df_train['split'] = 'train'

        df_dev['idx'] = 'dev_' + df_dev['idx'].astype(str)
        df_dev['split'] = 'dev'

        # We take as a test set valid split, cause BoolQ is SuperGLUE benchmark dataset and the real
        # test split has no ground truth labels
        df_test = pd.read_json(self.data_dir / 'val.jsonl', lines=True)
        df_test['idx'] = 'test_' + df_test['idx'].astype(str)
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df = df.rename(columns={'idx': 'text_id'})

        texts_df = df.loc[:, ['text_id', 'question', 'passage', 'split']]
        annotations_df = df.loc[:, ['text_id', 'label']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
