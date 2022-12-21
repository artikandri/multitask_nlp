from typing import List, Tuple

import pandas as pd
import ast

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import FACQA_QA_FACTOID_ITB_DATA


class FacqaQaFactoidItbDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = FACQA_QA_FACTOID_ITB_DATA
        self.annotation_column = ['seq_label']
        self.text_column = 'question'
        self.text_2_column = 'passage'

        self.train_split_names = ['train']
        self.val_split_names = ['valid']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [len(label_map) for label_map in self.label_maps]

    @property
    def task_name(self):
        return "facqa_qa-factoid-itb"

    @property
    def task_type(self):
        return 'sequence labeling'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_ids, texts, texts_2, labels, splits = self._get_data_from_split_files()
        self.data = pd.DataFrame({
            'text_id': text_ids,
            self.text_column: texts,
            self.text_2_column: texts_2,
            'split': splits,
        })

        label_map = {label: i for i, label in enumerate(self.get_labels())}
        labels = [[label_map[t] for t in tag[0]] for tag in labels]

        self.annotations = pd.DataFrame({
            'text_id': text_ids,
            'annotator_id': 0,
            'ner_tags': labels
        })
        self.label_maps = [label_map]

    def get_labels(self) -> List[str]:
        return ["O", "B", "I"]
    
    def _parse_row_columns(self, row, columns):
        for column in columns:
            row[column] = ast.literal_eval(row[column])
        return row
        
    def _parse_list(self, df, columns):
        df = df.apply(lambda row: self._parse_row_columns(row, columns), axis=1)
        return df

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        text_ids, texts, texts_2, labels, splits = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for index, split_name in enumerate(['train_preprocess', 'valid_preprocess', 'test_preprocess']):
            df = pd.read_csv(self.data_dir / f'{split_name}.csv')
            columns = df.columns
            df = self._parse_list(df, columns)
            df['text_id'] =  str(index)+ "_" + df.index.astype(str)
            df['splits'] = split_name
            text_ids = pd.concat([text_ids, df['text_id']])
            texts = pd.concat([texts, df['question']])
            texts_2 = pd.concat([texts_2, df['passage']])
            labels = pd.concat([labels, df['seq_label']])
            splits = pd.concat([splits, df['splits']])

        return text_ids.values.tolist(), texts.values.tolist(), texts_2.values.tolist(), labels.values.tolist(), splits.values.tolist()
