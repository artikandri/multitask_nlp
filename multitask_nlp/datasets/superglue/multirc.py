from typing import List, Tuple

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import MULTIRC_DATA


class MultiRCDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = MULTIRC_DATA
        self.annotation_column = ['label']
        self.text_column = 'question_answer'
        self.text_2_column = 'passage'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2]

    @property
    def task_name(self):
        return "MultiRC"

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
        df_list = []
        for split in ['train', 'dev', 'test']:
            df = pd.read_csv(self.data_dir / f'{split}_data.csv')
            df['split'] = split
            df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)
        df = df.rename(columns={'id': 'text_id'})

        texts_df =  df.loc[:, ['text_id', 'question_answer', 'passage', 'split']]
        annotations_df = df.loc[:, ['text_id', 'label']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
