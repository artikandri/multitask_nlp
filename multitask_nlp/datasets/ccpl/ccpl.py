import os
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.nkjp_pos_settings import ANNOTATION_COLUMNS, IOB_LABELS_MAPPING
from multitask_nlp.settings import CCPL_DIR
from multitask_nlp.utils.nkjp_pos_tagging import ParsedTag
from multitask_nlp.utils.iob_tagging import to_iob_tag

DEFAULT_SPLITS = [0.8, 0.1, 0.1]


class CCPL_DataModule(BaseDataModule):
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

        self.data_dir = CCPL_DIR
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
        return "CCPL"

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
        files_path = self.data_dir / 'anonimizacja_xml_out_ver'

        files = os.listdir(files_path)
        files = list(filter(lambda x: x.split('.')[-1] == 'xml', files))

        all_documents_data = []

        for f_name in files:
            try:
                data = self._read_xml_file(files_path / f_name)
                all_documents_data.append(data)
            except ET.ParseError:
                pass

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
    def _read_xml_file(filepath: str):
        tree = ET.parse(filepath)
        root = tree.getroot()
        chunk_node = root[0]

        if len(root) > 1:
            print('Warning')

        data = []
        for sentence_chunk in chunk_node.iter('sentence'):
            replace_previous = False

            sentence_tokens = []
            sentence_tags = []

            for token in sentence_chunk.iter('tok'):
                orth = token[0].text
                if orth is not None and len(orth) > 0:
                    lex_tags = [descendant for descendant in token.iter('lex')]
                    if len(lex_tags) > 0:
                        for lex_tag in lex_tags:
                            if 'disamb' in lex_tag.attrib:
                                pos_tag = lex_tag[1].text
                            else:
                                print('No disamb found.')
                                print(filepath)
                    else:
                        pos_tag = 'ign'

                    grammatic_class = pos_tag.split(':')[0]
                    if grammatic_class == '@www':
                        pos_tag = 'subst:sg:nom:n'

                    if grammatic_class != 'blank':
                        sentence_tokens.append(orth)
                        sentence_tags.append(pos_tag)
                    else:
                        if len(sentence_tokens) == 0 and len(data) > 0:
                            _, prev_sentence_tokens, prev_sentence_tags = data[-1]
                            sentence_tokens = prev_sentence_tokens
                            sentence_tags = prev_sentence_tags
                            replace_previous = True

                        sentence_tokens[-1] = sentence_tokens[-1] + orth

            text = ' '.join(sentence_tokens)
            if replace_previous:
                data[-1] = (text, sentence_tokens, sentence_tags)
            else:
                data.append((text, sentence_tokens, sentence_tags))

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
