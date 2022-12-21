import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import ASPECT_EMO_DIR
from multitask_nlp.utils.file_loading import read_lines_from_txt_file

DEFAULT_SPLITS = [0.7, 0.15, 0.15]


class ApectemoDataModule(BaseDataModule):
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

        self.data_dir = ASPECT_EMO_DIR
        self.split_sizes = split_sizes
        self.annotation_column = ['sentiment']
        self.text_column = 'text'
        self.tokens_column = 'tokens'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [len(label_map) for label_map in self.label_maps]

    @property
    def task_name(self):
        return "Aspectemo"

    @property
    def task_type(self):
        return 'sequence labeling'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_ids, texts, X_tokens, y = self._get_data_from_split_files()

        self.data = pd.DataFrame({
            'text_id': text_ids,
            self.text_column: texts,
            self.tokens_column: X_tokens
        })
        self.data[self.tokens_column] = self.data[self.tokens_column].to_numpy()

        flat_tags = [tag for tags in y for tag in tags]
        unique_tags = sorted(list(set(flat_tags)))
        label_map = {label: i for i, label in enumerate(unique_tags)}

        labels = [[label_map[t] for t in tags] for tags in y]
        self.annotations = pd.DataFrame({
            'text_id': text_ids,
            'annotator_id': 0,
            'sentiment': labels
        })

        self.label_maps = [label_map]
        self._assign_splits()

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        text_ids, texts, X_tokens, y = [], [], [], []
        documents_data_path = self.data_dir / 'documents'

        files = os.listdir(self.data_dir / 'documents')
        files = list(filter(lambda x: x.split('.')[1] == 'conll', files))

        for f_name in files:
            lines = read_lines_from_txt_file(documents_data_path / f_name)
            lines = [l for l in lines if l != '']
            rows = [l.split('\t') for l in lines][1:]

            tokens = [r[2] for r in rows]
            tags = [r[6].split(':')[0] for r in rows]
            text = ' '.join(tokens)

            text_ids.append(f_name.split('.')[0])
            texts.append(text)
            X_tokens.append(tokens)
            y.append(tags)

        return text_ids, texts, X_tokens, y

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
