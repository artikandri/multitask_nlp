from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import ALLEGRO_DATA

DEFAULT_RANDOM = 42


class AllegroReviewsDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = ALLEGRO_DATA
        self.annotation_column = ['rating']
        self.text_column = 'text'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [1]

    @property
    def task_name(self):
        return "AllegroReviews"

    @property
    def task_type(self):
        return 'regression'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_df, annotations_df = self._get_data_from_split_files()
        self.data = text_df
        self.annotations = annotations_df
        self.normalize_labels()

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        df_train_raw = pd.read_table(self.data_dir / 'train.tsv', delimiter=r"\t", engine='python')
        df_train_raw = df_train_raw.dropna(subset=['rating'])
        df_train, df_dev = train_test_split(
            df_train_raw, test_size=1000,
            stratify=df_train_raw['rating'],
            random_state=DEFAULT_RANDOM
        )

        df_train = df_train.reset_index(drop=True)
        df_train['text_id'] = 'train_' + df_train.index.astype(str)
        df_train['split'] = 'train'

        df_dev = df_dev.reset_index(drop=True)
        df_dev['text_id'] = 'dev_' + df_dev.index.astype(str)
        df_dev['split'] = 'dev'

        # We take as a test set valid split, cause Allegro is GLUE benchmark dataset and the real test
        # split has no ground truth labels
        df_test = pd.read_table(self.data_dir / 'dev.tsv', delimiter=r"\t", engine='python')
        df_test = df_test.dropna(subset=['rating']).reset_index(drop=True)
        df_test['text_id'] = 'test_' + df_test.index.astype(str)
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        texts_df = df.loc[:, ['text_id', 'text', 'split']]
        annotations_df = df.loc[:, ['text_id', 'rating']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
