from enum import IntEnum
from typing import List, Tuple

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import SCITAIL_DATA


class Labels(IntEnum):
    NEUTRAL = 0
    ENTAILS = 1


_CLASS_MAPPING = {
    "neutral": Labels.NEUTRAL,
    "entails": Labels.ENTAILS
}


class SciTailDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = SCITAIL_DATA
        self.annotation_column = ['label']
        self.text_column = 'premise'
        self.text_2_column = 'hypothesis'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2]

    @property
    def task_name(self):
        return "SciTail"

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
        tsv_data_path = self.data_dir / 'tsv_format'

        df_list = []
        for split in ['train', 'dev', 'test']:
            df = pd.read_csv(tsv_data_path / f'scitail_1.0_{split}.tsv', sep='\t',
                             names=['premise', 'hypothesis', 'label'])
            df['split'] = split
            df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)
        df['text_id'] = df.index

        texts_df = df.loc[:, ['text_id', 'premise', 'hypothesis', 'split']]
        annotations_df = df.loc[:, ['text_id', 'label']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
