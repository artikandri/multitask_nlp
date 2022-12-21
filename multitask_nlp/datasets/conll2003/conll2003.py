from typing import List, Tuple

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import CONLL2003_DATA


class Conll2003DataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = CONLL2003_DATA
        self.annotation_column = ['ner_tags']
        self.text_column = 'text'
        self.tokens_column = 'tokens'

        self.train_split_names = ['train']
        self.val_split_names = ['valid']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [len(label_map) for label_map in self.label_maps]

    @property
    def task_name(self):
        return "Conll2003"

    @property
    def task_type(self):
        return 'sequence labeling'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_ids, texts, texts_tokens, labels, splits = self._get_data_from_split_files()

        self.data = pd.DataFrame({
            'text_id': text_ids,
            self.text_column: texts,
            self.tokens_column: texts_tokens,
            'split': splits,
        })
        self.data[self.tokens_column] = self.data[self.tokens_column].to_numpy()

        label_map = {label: i for i, label in enumerate(self.get_labels())}
        labels = [[label_map[t] for t in tags] for tags in labels]

        self.annotations = pd.DataFrame({
            'text_id': text_ids,
            'annotator_id': 0,
            'ner_tags': labels
        })
        self.label_maps = [label_map]

    def get_labels(self) -> List[str]:
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        text_ids, texts, texts_tokens, labels, splits = [], [], [], [], []
        text_id = 1

        for split_name in ['train', 'valid', 'test']:
            data = self._read_file(self.data_dir / f'{split_name}.txt')
            for sentence, sentence_tokens, label in data:
                text_ids.append(text_id)
                texts.append(sentence)
                texts_tokens.append(sentence_tokens)
                labels.append(label)
                splits.append(split_name)

                text_id += 1

        return text_ids, texts, texts_tokens, labels, splits

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
