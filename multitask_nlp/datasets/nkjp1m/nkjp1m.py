from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.nkjp_pos_settings import ANNOTATION_COLUMNS, IOB_LABELS_MAPPING
from multitask_nlp.settings import NKJP1M_DIR
from multitask_nlp.utils.nkjp_pos_tagging import ParsedTag
from multitask_nlp.utils.iob_tagging import to_iob_tag

DEFAULT_SPLITS = [0.8, 0.1, 0.1]


class NKJP1M_DataModule(BaseDataModule):
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

        self.data_dir = NKJP1M_DIR
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
        return "NKJP1M"

    @property
    def task_type(self):
        return 'sequence labeling'

    @property
    def task_category(self):
        return 'pl pos tagging'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_ids, texts, texts_tokens, tags_per_categories, splits = self._get_data_from_file()

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

    def _get_data_from_file(self) -> Tuple[List, ...]:
        filepath = self.data_dir / 'nkjp1m-1.2-xces-xml'
        all_documents_data = self._read_xml_file(filepath)

        document_splits = self._get_splits(all_documents_data)

        text_ids, texts, texts_tokens, tags, splits = [], [], [], [], []

        for document_id, (document_data, document_split) in enumerate(
            zip(all_documents_data, document_splits), 1):
            document_sentences = []
            document_tokens = []
            document_tags = []

            for sentence_id, sentence_data in enumerate(document_data, 1):
                (sentence, sentence_tokens, sentence_tags) = sentence_data
                document_sentences.append(sentence)
                document_tokens += sentence_tokens
                document_tags += sentence_tags

            texts.append(' '.join(document_sentences))
            texts_tokens.append(document_tokens)
            tags.append(document_tags)
            text_ids.append(f"{document_id}")
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
    def _read_xml_file(filepath: str):
        tree = ET.parse(filepath)
        root = tree.getroot()
        chunk_list_root = root[0]

        all_documents_data = []

        for paragraph_chunk in chunk_list_root:
            pragraph_data = []
            for sentence_chunk in paragraph_chunk:
                sentence_tokens = []
                sentence_tags = []
                for token in sentence_chunk.iter('tok'):
                    orth = token[0].text
                    sentence_tokens.append(orth)

                    pos_tags = [descendant for descendant in token.iter('lex')]
                    for pos_tag in pos_tags:
                        if 'disamb' in pos_tag.attrib:
                            sentence_tags.append(pos_tag[1].text)

                text = ' '.join(sentence_tokens)
                pragraph_data.append((text, sentence_tokens, sentence_tags))

            all_documents_data.append(pragraph_data)

        return all_documents_data

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
