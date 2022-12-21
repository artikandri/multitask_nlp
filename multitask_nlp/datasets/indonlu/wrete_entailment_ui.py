from enum import IntEnum
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import  WRETE_ENTAILMENT_DATA

DEFAULT_RANDOM = 42

class Labels(IntEnum):
    NOT_ENTAILMENT = 0
    ENTAILMENT = 1


_CLASS_MAPPING = {
    "NotEntail": Labels.NOT_ENTAILMENT,
    "Entail_or_Paraphrase": Labels.ENTAILMENT
}


class WreteEntailmentUiDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = WRETE_ENTAILMENT_DATA
        self.annotation_column = ['label']
        self.text_column = 'sent_A'
        self.text_2_column = 'sent_B'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2]*1

    @property
    def task_name(self):
        return "wrete_entailment-ui"

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
        df_train = pd.read_table(self.data_dir / 'train_preprocess.csv', delimiter=",", engine='python')
        df_train['index'] = 'train_' + df_train.index.astype(str)
        df_train['split'] = 'train'

        df_dev = pd.read_table(self.data_dir / 'valid_preprocess.csv', delimiter=",", engine='python')
        df_dev['index'] = 'dev_' + df_dev.index.astype(str)
        df_dev['split'] = 'dev'

        df_test = pd.read_csv(self.data_dir / 'test_preprocess.csv', delimiter=",", engine='python')
        df_test['index'] = 'test_' + df_test.index.astype(str)
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df = df.rename(columns={'index': 'text_id'})
        
        texts_df = df.loc[:, ['text_id', 'sent_A', 'sent_B', 'split']]
        annotations_df = df.loc[:, ['text_id', 'label']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
