import os
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.nkjp_pos_settings import ANNOTATION_COLUMNS, IOB_LABELS_MAPPING
from multitask_nlp.settings import POLEVAL18_POS_DIR
from multitask_nlp.utils.nkjp_pos_tagging import ParsedTag
from multitask_nlp.utils.iob_tagging import to_iob_tag

DEFAULT_SPLITS = [0.8, 0.1, 0.1]


class PolEval18_POS_DataModule(BaseDataModule):
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

        self.data_dir = POLEVAL18_POS_DIR
        self.split_sizes = split_sizes
        self.annotation_column = ANNOTATION_COLUMNS
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
        return "PolEval18_POS"

    @property
    def task_type(self):
        return 'sequence labeling'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    @property
    def task_category(self):
        return 'pl pos tagging'

    def prepare_data(self) -> None:
        text_ids, texts, texts_tokens, tags_per_categories, splits = self._get_data_from_files()

        self.data = pd.DataFrame({
            'text_id': text_ids,
            self.text_column: texts,
            self.tokens_column: texts_tokens,
            'split': splits
        })
        self.data[self.tokens_column] = self.data[self.tokens_column].to_numpy()

        self.label_maps = IOB_LABELS_MAPPING
        encoded_tags_per_categories = []
        for tags_per_category, label_map in zip(tags_per_categories, self.label_maps):
            labels = [[label_map[tag] for tag in tags] for tags in tags_per_category]
            encoded_tags_per_categories.append(labels)

        data_for_annotations_df = {
            'text_id': text_ids,
            'annotator_id': 0
        }
        for annotation_column, labels in zip(ANNOTATION_COLUMNS, encoded_tags_per_categories):
            data_for_annotations_df[annotation_column] = labels

        self.annotations = pd.DataFrame(data_for_annotations_df)

    def _get_data_from_files(self) -> Tuple[List, ...]:

        files = os.listdir(self.data_dir)
        files = list(filter(lambda x: x.split('.')[1] == 'iob', files))

        all_documents_data = []

        for f_name in files:
            data = self._read_file(self.data_dir / f_name)
            all_documents_data.append(data)

        document_splits = self._get_splits(all_documents_data)

        text_ids, texts, texts_tokens, tags, splits = [], [], [], [], []

        for document_data, document_split, f_name in zip(all_documents_data, document_splits,
                                                         files):
            for sentence_id, sentence_data in enumerate(document_data, 1):
                (sentence, sentence_tokens, sentence_tags) = sentence_data

                texts.append(sentence)
                texts_tokens.append(sentence_tokens)
                tags.append(sentence_tags)
                text_ids.append(f"{f_name.split('.')[0]}_{sentence_id}")
                splits.append(document_split)

        assert all([len(tokens) == len(texts_tags) for tokens, texts_tags
                    in zip(texts_tokens, tags)])

        parsed_tags = [[ParsedTag.from_tag(t).to_list() for t in text_tags] for text_tags in tags]

        parsed_tags = [
            [[to_iob_tag(t) for t in category_tags] for category_tags in text_tags]
            for text_tags in parsed_tags
        ]
        tags_per_categories = [list(z) for z in zip(*[[list(z) for z in zip(*sent_parsed_tags)] for
                                                      sent_parsed_tags in parsed_tags])]

        return text_ids, texts, texts_tokens, tags_per_categories, splits

    @staticmethod
    def _read_file(filepath: str) -> List[Tuple[str, List[str], List[str]]]:
        data = []
        sentence_tokens = []
        tags = []

        f = open(filepath, encoding='UTF-8')
        for i, line in enumerate(f, 1):
            if not line.strip() or len(line) == 0 or \
                line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence_tokens) > 0:
                    sentence = ' '.join(sentence_tokens)
                    data.append((sentence, sentence_tokens, tags))
                    sentence_tokens = []
                    tags = []
                continue

            splits = line.split('\t')
            assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
            word, tag = splits[0], splits[2]
            sentence_tokens.append(word.strip())
            tags.append(tag.strip())

        if len(sentence_tokens) > 0:
            sentence = ' '.join(sentence_tokens)
            data.append((sentence, sentence_tokens, tags))

        f.close()
        return data

    def _get_splits(self, data: List[Any]) -> List[str]:
        train_ratio, valid_ratio, test_ration = self.split_sizes
        train_idx = int(train_ratio * len(data))
        valid_idx = int(valid_ratio * len(data)) + train_idx

        indexes = np.arange(len(data))
        np.random.shuffle(indexes)

        splits = np.array([''] * len(data), dtype=object)
        splits[indexes[:train_idx]] = 'train'
        splits[indexes[train_idx:valid_idx]] = 'dev'
        splits[indexes[valid_idx:]] = 'test'

        return splits.tolist()
