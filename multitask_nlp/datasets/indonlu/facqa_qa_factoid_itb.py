from typing import List, Tuple
from enum import IntEnum

import pandas as pd
import ast

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import FACQA_QA_FACTOID_ITB_DATA

class Labels(IntEnum):
    I = 0
    O = 1
    B = 2

_CLASS_MAPPING = {
    "I": Labels.I,
    "O": Labels.O,
    "B": Labels.B,
}

class FacqaQaFactoidItbDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = FACQA_QA_FACTOID_ITB_DATA
        self.annotation_column = ['labels']
        self.text_column = 'passage_text'
        self.tokens_column = 'tokens'
        self.label_maps = [_CLASS_MAPPING]

        self.train_split_names = ['train']
        self.val_split_names = ['valid']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [len(label_map) for label_map in self.label_maps] * 1

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
        df = self._get_data_from_split_files()

        labels = df['seq_label'].apply(lambda row : [i.strip() for i in row[1:-1]])
        labels  = df['seq_label'].values.tolist()
        labels = list(map(lambda label: list((pd.Series(label)).map(_CLASS_MAPPING)), labels))

        self.data = pd.DataFrame({
            'text_id': df['text_id'],
            self.text_column: df['passage_text'],
            self.tokens_column: df['passage'],
            'split': df['splits'],
        })
        self.annotations = pd.DataFrame({
            'text_id': df['text_id'],
            'annotator_id': 0,
            self.annotation_column: labels
        })
    
    def _parse_row_columns(self, row, columns):
        for column in columns:
            row[column] = ast.literal_eval(row[column])
        return row
        
    def _parse_list(self, df, columns):
        df = df.apply(lambda row: self._parse_row_columns(row, columns), axis=1)
        return df

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        master_df = pd.DataFrame()
        for index, split_name in enumerate(['train_preprocess', 'valid_preprocess', 'test_preprocess']):
            split = split_name.split("_")[0]
            df = pd.read_csv(self.data_dir / f'{split_name}.csv')
            columns = df.columns
            df = self._parse_list(df, columns)
            df['question_text'] = df['question'].apply(lambda row: (" ".join(row)))
            df['passage_text'] = df['passage'].apply(lambda row: (" ".join(row)))
            df['text_id'] =  split+"_"+str(index)+ "_" + df.index.astype(str)
            df['splits'] = split
            master_df = pd.concat([master_df, df])

        return master_df
