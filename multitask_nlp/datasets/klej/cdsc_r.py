from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import CDSC_R_DATA

DEFAULT_RANDOM = 42


class CDSC_R_DataModule(BaseDataModule):
    def __init__(
        self,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = CDSC_R_DATA
        self.annotation_column = ['relatedness_score']
        self.text_column = 'sentence_A'
        self.text_2_column = 'sentence_B'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

        self.normalize = normalize

    @property
    def class_dims(self):
        return [1]

    @property
    def task_name(self):
        return "CDSC-R"

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
        df_train_raw = pd.read_table(self.data_dir / 'train.tsv', delimiter=r"\t", engine='python')
        df_train, df_dev = train_test_split(
            df_train_raw, test_size=1000,
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
        annotations_df = df.loc[:, ['text_id', 'relatedness_score']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
