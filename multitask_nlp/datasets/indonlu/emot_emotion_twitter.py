from typing import List, Tuple

import pandas as pd
import numpy as np

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import EMOT_EMOTION_TWITTER_DATA

DEFAULT_RANDOM = 42
DEFAULT_SPLITS = [0.7, 0.15, 0.15]
ANNOTATION_COLUMNS = ['love', 'fear', 'anger', 'sadness', 'happy']

class EmotEmotionTwitterDataModule(BaseDataModule):
    def __init__(
        self,
        split_sizes: List[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if split_sizes is None:
            split_sizes = DEFAULT_SPLITS
        assert len(split_sizes) == 3 and sum(split_sizes) == 1 and all(
            0 <= s <= 1 for s in split_sizes)

        self.split_sizes = split_sizes
        self.data_dir = EMOT_EMOTION_TWITTER_DATA
        self.annotation_column = ANNOTATION_COLUMNS
        self.text_column = 'text'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2]*5 

    @property
    def task_name(self):
        return "emot_emotion-twitter"

    @property
    def task_type(self):
        return 'classification'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def _assign_splits(self):
        data = self.data

        train_ratio, valid_ratio, test_ration = self.split_sizes
        train_idx = int(train_ratio * len(data))
        valid_idx = int(valid_ratio * len(data)) + train_idx

        indexes = np.arange(len(data.index))
        np.random.shuffle(indexes)

        data['split'] = ''
        data.iloc[indexes[:train_idx], data.columns.get_loc('split')] = 'train'
        data.iloc[indexes[train_idx:valid_idx], data.columns.get_loc('split')] = 'dev'
        data.iloc[indexes[valid_idx:], data.columns.get_loc('split')] = 'test'

        self.data = data

    def prepare_data(self) -> None:
        text_df, annotations_df = self._get_data_from_split_files()
        self.data = text_df
        self.annotations = annotations_df
        self._assign_splits()

    def _set_label(self, row, column):
        return 1 if row['label'] == column else 0
    
    def _map_labels(self, df):
        for column in ANNOTATION_COLUMNS:
            df[column] = df.apply(lambda row: self._set_label(row, column), axis=1)
        return df

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        df_train = pd.read_csv(self.data_dir / 'train_preprocess.csv')
        df_test = pd.read_csv(self.data_dir / 'test_preprocess.csv')
        df_dev = pd.read_csv(self.data_dir / 'valid_preprocess.csv')

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df = self._map_labels(df)

        df['text_id'] = df.index
        df['text'] = df['tweet']

        texts_df = df.loc[:, ['text_id', 'text']]
        annotations_df = df
        annotations_df.loc[:, 'annotator_id'] = 0

        return texts_df, annotations_df
