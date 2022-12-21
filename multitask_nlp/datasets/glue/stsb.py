from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import STS_B_DATA

DEFAULT_RANDOM = 42


class STS_B_DataModule(BaseDataModule):
    def __init__(
        self,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = STS_B_DATA
        self.annotation_column = ['label']
        self.text_column = 'sentence1'
        self.text_2_column = 'sentence2'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

        self.normalize = normalize

    @property
    def class_dims(self):
        return [1]

    @property
    def task_name(self):
        return "STS-B"

    @property
    def task_type(self):
        return 'regression'

    @property
    def texts_clean(self):
        return (self.data[self.text_column] + ' ' + self.data[self.text_2_column]).to_list()

    def prepare_data(self) -> None:
        text_df, annotations_df = self._get_data_from_split_files()

        self.data = text_df
        self.annotations = annotations_df

        if self.normalize:
            self.normalize_labels()

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        df_train_raw = pd.read_csv(self.data_dir / 'data_train_raw.csv')

        df_train, df_dev = train_test_split(df_train_raw, test_size=1500,
                                            random_state=DEFAULT_RANDOM)
        df_train['idx'] = 'train_' + df_train['idx'].astype(str)
        df_train['split'] = 'train'

        df_dev['idx'] = 'dev_' + df_dev['idx'].astype(str)
        df_dev['split'] = 'dev'

        # We take as a test set valid split, cause STS-B is GLUE benchmark dataset and the real test
        # split has no ground truth labels
        df_test = pd.read_csv(self.data_dir / 'data_validation_raw.csv')
        df_test['idx'] = 'test_' + df_test['idx'].astype(str)
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df = df.rename(columns={'idx': 'text_id'})

        texts_df = df.loc[:, ['text_id', 'sentence1', 'sentence2', 'split']]
        annotations_df = df.loc[:, ['text_id', 'label']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
