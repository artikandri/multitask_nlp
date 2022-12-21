from enum import IntEnum
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import RTE_DATA, SCITAIL_DATA

DEFAULT_RANDOM = 42


class Labels(IntEnum):
    NOT_ENTAILMENT = 0
    ENTAILMENT = 1


_CLASS_MAPPING = {
    "not_entailment": Labels.NOT_ENTAILMENT,
    "entailment": Labels.ENTAILMENT
}


class RTE_DataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = RTE_DATA
        self.annotation_column = ['label']
        self.text_column = 'sentence1'
        self.text_2_column = 'sentence2'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2]

    @property
    def task_name(self):
        return "RTE"

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
        df_train_raw = pd.read_table(self.data_dir / 'train.tsv', delimiter=r"\t", engine='python')

        df_train, df_dev = train_test_split(df_train_raw, test_size=300,
                                            random_state=DEFAULT_RANDOM)
        df_train['index'] = 'train_' + df_train['index'].astype(str)
        df_train['split'] = 'train'

        df_dev['index'] = 'dev_' + df_dev['index'].astype(str)
        df_dev['split'] = 'dev'

        # We take as a test set valid split, cause RTE is GLUE benchmark dataset and the real test
        # split has no ground truth labels
        df_test = pd.read_csv(self.data_dir / 'dev.tsv', delimiter=r"\t", engine='python')
        df_test['index'] = 'test_' + df_test['index'].astype(str)
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df = df.rename(columns={'index': 'text_id'})

        texts_df = df.loc[:, ['text_id', 'sentence1', 'sentence2', 'split']]
        annotations_df = df.loc[:, ['text_id', 'label']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
