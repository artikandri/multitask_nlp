from enum import IntEnum
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import POLEMO_IN_DATA, POLEMO_OUT_DATA

DEFAULT_RANDOM = 42


class Labels(IntEnum):
    AMBIVALENT = 0
    NEGATIVE = 1
    POSITIVE = 2
    NEUTRAL = 3


_SENTIMENT_MAPPING = {
    "__label__meta_amb": Labels.AMBIVALENT,
    "__label__meta_minus_m": Labels.NEGATIVE,
    "__label__meta_plus_m": Labels.POSITIVE,
    "__label__meta_zero": Labels.NEUTRAL,
}


class KlejPolemoDataModule(BaseDataModule):
    """Datamodule for KLEJ PolEmo dataset.

    KLEJ has two versions of PolEmo. One for which test and train data are from the same domain
    (INSIDE) and another one where test data are from different domain than train data (OUTSIDE).
    """
    def __init__(
        self,
        domain_inside: bool = True,
        **kwargs,
    ):
        """Initializes KLEJ PolEmo datamodule.

        Args:
            domain_inside (bool, optional): Whether to use INSIDE or OUTSIDE version of dataset.
                Defaults to True, i.e., INSIDE variant.
            **kwargs ():
        """
        super().__init__(**kwargs)
        self.domain_inside = domain_inside

        if domain_inside:
            self.data_dir = POLEMO_IN_DATA
        else:
            self.data_dir = POLEMO_OUT_DATA

        self.annotation_column = ['target']
        self.text_column = 'sentence'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [4]

    @property
    def task_name(self):
        if self.domain_inside:
            return "KLEJ_PolEmo-IN"
        else:
            return "KLEJ_PolEmo-OUT"

    @property
    def task_type(self):
        return 'classification'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_df, annotations_df = self._get_data_from_split_files()
        self.data = text_df
        self.annotations = annotations_df
        self.annotations['target'].replace(_SENTIMENT_MAPPING, inplace=True)

    def _get_data_from_split_files(self) -> Tuple[List, ...]:
        df_train_raw = pd.read_table(self.data_dir / 'train.tsv', delimiter=r"\t", engine='python')
        df_train, df_dev = train_test_split(
            df_train_raw, test_size=700,
            stratify=df_train_raw['target'],
            random_state=DEFAULT_RANDOM
        )

        df_train = df_train.reset_index(drop=True)
        df_train['text_id'] = 'train_' + df_train.index.astype(str)
        df_train['split'] = 'train'

        df_dev = df_dev.reset_index(drop=True)
        df_dev['text_id'] = 'dev_' + df_dev.index.astype(str)
        df_dev['split'] = 'dev'

        # We take as a test set valid split, cause KLEJ PolEmo is KLEJ benchmark dataset and the real
        # test split has no ground truth labels
        df_test = pd.read_table(self.data_dir / 'dev.tsv', delimiter=r"\t", engine='python')
        df_test['text_id'] = 'test_' + df_test.index.astype(str)
        df_test['split'] = 'test'

        df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        texts_df = df.loc[:, ['text_id', 'sentence', 'split']]
        annotations_df = df.loc[:, ['text_id', 'target']]
        annotations_df.loc[:, 'annotator_id'] = 0
        return texts_df, annotations_df
