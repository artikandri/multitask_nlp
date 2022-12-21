from enum import IntEnum
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import CDSC_E_DATA

DEFAULT_RANDOM = 42


class Labels(IntEnum):
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2


_CLASS_MAPPING = {
    "CONTRADICTION": Labels.CONTRADICTION,
    "NEUTRAL": Labels.NEUTRAL,
    "ENTAILMENT": Labels.ENTAILMENT
}


class CDSC_E_DataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = CDSC_E_DATA
        self.annotation_column = ['entailment_judgment']
        self.text_column = 'sentence_A'
        self.text_2_column = 'sentence_B'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [3]

    @property
    def task_name(self):
        return "CDSC-E"

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
        self.annotations['entailment_judgment'].replace(_CLASS_MAPPING, inplace=True)

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        df_train_raw = pd.read_table(self.data_dir / 'train.tsv', delimiter=r"\t", engine='python')
        df_train, df_dev = train_test_split(
            df_train_raw, test_size=1000,
            stratify=df_train_raw['entailment_judgment'],
            random_state=DEFAULT_RANDOM
        )

        df_train['pair_ID'] = 'train_' + df_train['pair_ID'].astype(str)
        df_train['split'] = 'train'

        df_dev['pair_ID'] = 'dev_' + df_dev['pair_ID'].astype(str)
        df_dev['split'] = 'dev'

        # We take as a test set valid split, cause CDSC is KLEJ benchmark dataset and the real test
        # split has no ground truth labels
        df_test = pd.read_table(self.data_dir / 'dev.tsv', delimiter=r"\t", engine='python')
        df_test['pair_ID'] = 'test_' + df_test['pair_ID'].astype(str)
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df = df.rename(columns={'pair_ID': 'text_id'})
        texts_df = df.loc[:, ['text_id', 'sentence_A', 'sentence_B', 'split']]
        annotations_df = df.loc[:, ['text_id', 'entailment_judgment']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
