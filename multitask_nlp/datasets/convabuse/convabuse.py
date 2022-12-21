from enum import IntEnum

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import CONVABUSE_DATA


class Labels(IntEnum):
    VERY_STRONGLY_ABUSIVE = 0
    STRONGLY_ABUSIVE = 1
    MILDLY_ABUSIVE = 2
    AMBIGUOUS = 3
    NOT_ABUSIVE = 4


_IS_ABUSE_MAPPING = {
    -3: Labels.VERY_STRONGLY_ABUSIVE,
    -2: Labels.STRONGLY_ABUSIVE,
    -1: Labels.MILDLY_ABUSIVE,
    0: Labels.AMBIGUOUS,
    1: Labels.NOT_ABUSIVE,
}


class ConvAbuseDataModule(BaseDataModule):
    def __init__(
        self,
        binary_major_voting: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = CONVABUSE_DATA
        self.annotation_column = ['is_abuse']
        self.text_column = 'text_agent'
        self.text_2_column = 'text_user'

        self.train_split_names = ['train']
        self.val_split_names = ['valid']
        self.test_split_names = ['test']

        self.binary_major_voting = binary_major_voting

    @property
    def class_dims(self):
        if self.binary_major_voting:
            return [2]
        return [5]

    @property
    def task_name(self):
        return "ConvAbuse"

    @property
    def task_type(self):
        return 'classification'

    @property
    def texts_clean(self):
        return (self.data[self.text_column] + ' ' + self.data[self.text_2_column]).to_list()

    def prepare_data(self) -> None:
        text_data_df_list = []
        for split in ['train', 'valid', 'test']:
            text_data_split = pd.read_csv(self.data_dir / f'text_data_{split}.csv')
            text_data_split['split'] = split
            text_data_df_list.append(text_data_split)

        self.data = pd.concat(text_data_df_list, ignore_index=True)
        self.data = self.data.rename(columns={'agent': 'text_agent', 'user': 'text_user'})

        if self.binary_major_voting:
            self.annotations = pd.concat(
                [pd.read_csv(self.data_dir / f'binary_abuse_mv_{split}.csv')
                 for split in ['train', 'valid', 'test']]
            )
            self.annotations['annotator_id'] = 0
        else:
            self.annotations = pd.concat(
                [pd.read_csv(self.data_dir / f'annotations_{split}.csv')
                 for split in ['train', 'valid', 'test']], ignore_index=True
            )
            self.annotations['is_abuse'].replace(_IS_ABUSE_MAPPING, inplace=True)
