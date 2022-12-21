from typing import List, Tuple
from enum import IntEnum

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import MULTINLI_DATA

DEFAULT_RANDOM = 42
DEFAULT_SPLITS = [0.6, 0.2, 0.2]

class Labels(IntEnum):
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2


_CLASS_MAPPING = {
    "contradiction": Labels.CONTRADICTION,
    "neutral": Labels.NEUTRAL,
    "entailment": Labels.ENTAILMENT
}


class MultiNLIDataModule(BaseDataModule):
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

        self.data_dir = MULTINLI_DATA
        self.annotation_column = ['label']
        self.text_column = 'premise'
        self.text_2_column = 'hypothesis'
        self.split_sizes = split_sizes

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [3] * 1

    @property
    def task_name(self):
        return "MultiNLI"

    @property
    def task_type(self):
        return 'classification'

    @property
    def texts_clean(self):
        return (self.data[self.text_column] + ' ' + self.data[self.text_2_column]).to_list()

    def prepare_data(self) -> None:
        self.data = self._read_file(self.data_dir / 'multinli_1.0_train.txt')
        self.annotations = self._read_file(self.data_dir / 'multinli_1.0_train.txt').dropna()
        self.annotations['label'].replace(_CLASS_MAPPING, inplace=True)

    def _read_file(self, filename: str) -> List[Tuple[str, List[str], List[str]]]:
        data = []
        sentence_tokens = []
        label = []

        f = open(filename)
        for i, line in enumerate(f, 1):
            if not line.strip() or len(line) == 0 or line.startswith('-DOCSTART') \
                    or line[0] == "\n" or line[0] == '.':
                if len(sentence_tokens) > 0:
                    sentence = ' '.join(sentence_tokens)
                    data.append((sentence, sentence_tokens, label))
                    sentence_tokens = []
                    label = []
                continue

            splits = line.split()
            assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
            word, tag = splits[0], splits[-1]
            assert tag in self.get_labels(), "unknown tag {} in line {}".format(tag, i)
            sentence_tokens.append(word.strip())
            label.append(tag.strip())

        if len(sentence_tokens) > 0:
            sentence = ' '.join(sentence_tokens)
            data.append((sentence, sentence_tokens, label))

        f.close()
        return data