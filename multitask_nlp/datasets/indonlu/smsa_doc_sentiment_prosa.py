from typing import List, Tuple
from pathlib import Path

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import SMSA_DOC_SENTIMENT_PROSA_DATA

_CLASS_MAPPING = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}

class SmsaDocSentimentProsaDataModule(BaseDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = SMSA_DOC_SENTIMENT_PROSA_DATA
        self.text_column = 'text'
        self.annotation_column = 'sentiment'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [3] * 1

    @property
    def task_name(self):
        return "smsa_doc_sentiment_prosa"

    @property
    def task_type(self):
        return 'classification'

    @property
    def task_category(self):
        return 'id sentiment classification'

    def prepare_data(self) -> None:
        texts_df, annotations_df = self._get_data_from_split_files()
        self.data =  texts_df
        self.annotations = annotations_df

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        df_train = self._load_df(self.data_dir / 'train_preprocess.tsv')
        df_dev = self._load_df(self.data_dir / 'valid_preprocess.tsv')
        df_test = self._load_df(self.data_dir / 'test_preprocess.tsv')
        
        df_train['split'] = 'train'
        df_dev['split'] = 'dev'
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        df['annotator_id'] = 0
        df['text_id'] = df.index
        df['sentiment'].replace(_CLASS_MAPPING, inplace=True)

        texts_df = df.loc[:, ['text_id', 'text', 'split']]
        annotations_df = df.loc[:, ['text_id', 'sentiment', 'annotator_id']]
        return texts_df, annotations_df

    @staticmethod
    def _load_df(directory: Path) -> pd.DataFrame:
        df = pd.read_table(directory, delimiter=r"\t", engine='python', names=['text', 'sentiment'])
        return df
