import pandas as pd
import numpy as np

from typing import List, Tuple
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import INDONESIAN_EMOTION_DATASET_DIR

DEFAULT_RANDOM = 42
DEFAULT_SPLITS = [0.7, 0.15, 0.15]
ANNOTATION_COLUMNS = ['Love', 'Fear', 'Anger', 'Sad', 'Joy', 'Neutral']

class IndonesianEmotionDataModule(BaseDataModule):
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


        self.data_dir = INDONESIAN_EMOTION_DATASET_DIR
        self.split_sizes = split_sizes
        self.annotation_column = ANNOTATION_COLUMNS
        self.text_column = 'Tweet'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [2] * 6

    @property
    def task_name(self):
        return "indonesian_emotion"

    @property
    def task_type(self):
        return 'classification'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def _set_label(self, row, column):
        return 1 if row['Label'] == column else 0
    
    def _map_labels(self, df):
        for column in ANNOTATION_COLUMNS:
            df[column] = df.apply(lambda row: self._set_label(row, column), axis=1)
        return df

    def _combine_datasets(self):
        filenames = ['AngerData.csv', 'JoyData.csv', 'SadData.csv', 'FearData.csv', 'LoveData.csv', 'NeutralData.csv']
        combined_data = pd.concat([pd.read_csv(f'{self.data_dir}/{f}', sep=";", on_bad_lines='skip') for f in filenames ])
        return combined_data

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

    def _get_data(self, dataset):
        df = dataset
        df['text_id'] = df.index
        df['text'] = df['Tweet']
        df = df.drop(ANNOTATION_COLUMNS, axis=1)
        df = df.drop(['Label'], axis=1)
        return df

    def _get_annotations(self, dataset):
        df = dataset
        df['text_id'] = df.index
        df['annotator_id'] = 0
        df = df.drop(['Tweet', 'Label'], axis=1)
        return df
        
    def prepare_data(self) -> None:
        dataset = self._combine_datasets()
        dataset = self._map_labels(dataset)
        self.data = self._get_data(dataset)
        self.annotations = self._get_annotations(dataset)
        self._assign_splits()
