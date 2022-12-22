from enum import IntEnum
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import NERP_NER_PROSA_DATA

_CLASS_MAPPING = {
    "I-IND": 0,
    "I-PLC": 1,
    "I-EVT": 2,
    "I-PPL": 3,
    "I-FNB": 4,
    "O": 5,
    "B-IND": 6,
    "B-PLC": 7,
    "B-EVT": 8,
    "B-PPL": 9,
    "B-FNB": 10
}

class NerpNerProsaDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = NERP_NER_PROSA_DATA
        self.text_column = 'text'
        self.annotation_column = 'sentiment'
        self.label_maps = [_CLASS_MAPPING]
        self.tokens_column = 'tokens'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [11] * 1

    @property
    def task_name(self):
        return "nerp_ner_prosa"

    @property
    def task_type(self):
        return 'sequence labeling'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_ids, texts, tokens, labels, splits = self._get_data_from_split_files()
        self.data = pd.DataFrame({
            'text_id': text_ids,
            self.tokens_column: tokens,
            self.text_column: texts,
            'split': splits,
        })
        self.annotations = pd.DataFrame({
            'text_id': text_ids,
            'annotator_id': 0,
            self.annotation_column: labels
        })

    def _read_lines_from_txt_file(self, path) -> Tuple[List, ...]:
        texts = []
        tokens = []
        labels = []
        with open(path) as f:
            text_lines, label_lines = [], []
            for line in f:
                if line.startswith(" ") or line.startswith("\n"):
                    text_lines_joined = " ".join(text_lines)
                    label_lines = list((pd.Series(label_lines)).map(_CLASS_MAPPING))
                    texts.append(text_lines_joined)
                    tokens.append(text_lines)
                    labels.append(label_lines)
                    text_lines, label_lines = [], []
                    continue
                else:
                    if(len(line.split()) > 1):
                        text_lines.append(line.split()[0])
                        label_lines.append(line.split()[1])
        return texts, tokens, labels
        
    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        text_ids = list()
        texts = list()
        labels = list()
        tokens = list()
        splits = list()

        for split_name in ['train', 'valid', 'test']:
            path = str(self.data_dir)+"/"+split_name+"_preprocess.txt"
            split_texts, split_tokens, split_labels = self._read_lines_from_txt_file(path)

            split_text_ids = range(0, len(split_labels))
            split_text_ids = list(map(lambda n: split_name+"_"+str(n), split_text_ids))

            text_ids += split_text_ids
            texts += split_texts
            labels += split_labels
            tokens += split_tokens
            splits += [split_name] * len(split_text_ids)

        return text_ids, texts, tokens, labels, splits
