from enum import IntEnum
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import MULTIEMO_DIR
from multitask_nlp.utils.file_loading import read_lines_from_txt_file


class Labels(IntEnum):
    AMBIVALENT = 0
    NEGATIVE = 1
    POSITIVE = 2
    NEUTRAL = 3


_SENTIMENT_MAPPING = {
    "meta_amb": Labels.AMBIVALENT,
    "meta_minus_m": Labels.NEGATIVE,
    "meta_plus_m": Labels.POSITIVE,
    "meta_zero": Labels.NEUTRAL,
}

_AVAILABLE_DOMAINS = ['all']
_AVAILABLE_LANGUAGES = ['en', 'pl']


class MultiemoDataModule(BaseDataModule):
    def __init__(
        self,
        domain: str = 'all',
        language: str = 'en',
        **kwargs,
    ):
        super().__init__(**kwargs)
        if domain not in _AVAILABLE_DOMAINS:
            raise ValueError(f"Wrong domain. Possible domains are: {', '.join(_AVAILABLE_DOMAINS)}")

        if language not in _AVAILABLE_LANGUAGES:
            raise ValueError(f"Wrong language. Possible domains are: {', '.join(_AVAILABLE_LANGUAGES)}")

        self.data_dir = MULTIEMO_DIR
        self.annotation_column = ['sentiment']
        self.text_column = 'text'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

        self.domain = domain
        self.language = language

    @property
    def class_dims(self):
        return [4]

    @property
    def task_name(self):
        return "Multiemo"

    @property
    def task_type(self):
        return 'classification'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_ids, texts, labels, splits = self._get_data_from_split_files()

        self.data = pd.DataFrame({
            'text_id': text_ids,
            self.text_column: texts,
            'split': splits,
        })

        self.annotations = pd.DataFrame({
            'text_id': text_ids,
            'annotator_id': 0,
            'sentiment': labels
        })
        self.annotations['sentiment'].replace(_SENTIMENT_MAPPING, inplace=True)

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        text_ids = list()
        texts = list()
        labels = list()
        splits = list()

        for split_name in ['train', 'dev', 'test']:
            lines = read_lines_from_txt_file(self._get_split_data_path(split_name))

            split_text_ids = list()
            split_texts = list()
            split_labels = list()

            for i, line in enumerate(lines):
                text_id = split_name + '_' + str(i)
                split_line = line.split('__label__')
                text = split_line[0]
                label = split_line[1]

                split_text_ids.append(text_id)
                split_texts.append(text)
                split_labels.append(label)

            text_ids += split_text_ids
            texts += split_texts
            labels += split_labels
            splits += [split_name] * len(split_text_ids)

        return text_ids, texts, labels, splits

    def _get_domain_to_keep(self, split_name) -> List[str]:
        domains_to_keep = ['hotels', 'medicine', 'products', 'reviews']
        if self.domain != 'all':
            if self.domain.startswith('N'):
                considered_domain = self.domain.split('N')[1]
                if split_name == 'test':
                    domains_to_keep = [considered_domain]
                else:
                    domains_to_keep.remove(considered_domain)
            else:
                domains_to_keep = [self.domain]

        return domains_to_keep

    def _get_split_data_path(self, split_name: str) -> Path:
        filename = self.domain + '.text.' + split_name + '.' + self.language + '.txt'
        return self.data_dir / filename
