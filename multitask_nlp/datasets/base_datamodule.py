import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from typing import Any, Dict, List, Tuple, Optional

from multitask_nlp.datasets.base_dataset import BaseDataset


class BaseDataModule(LightningDataModule):
    """Base class for DataModule used in the project.

    It encompasses dataset with train, val and test dataloaders which can be utilized by
    Lightning Trainer API.

    Attributes:
        batch_size (int): Size of the batch.
        major_voting (bool): Flag indicating if major voting is applied (only needed for datasets
        with many annotations per text example).
        num_workers (int): Number of workers used by dataloaders.
    """
    @property
    def task_type(self) -> str:
        """str: Type of task, e.g., classification or regression."""
        raise NotImplementedError()

    @property
    def task_name(self) -> str:
        """str: Name of task."""
        raise NotImplementedError()

    @property
    def task_category(self) -> str:
        """str: Category of the task.

        By default, it is the name of task, but can be different in the
        case of various datasets which have the same ground truth structure, e.g., POS tags
        with NKJP POS tagging structure.
        """
        return self.task_name

    @property
    def class_dims(self) -> List[int]:
        """list of int: Indicates dimensionalities of subsequent class (categories) to predict.

        When there is only one class to predict, it is a single element list. For regression tasks,
        it will be a list of ones with length indicating how many dimensions are considered
        in a regression task.
        """
        raise NotImplementedError()

    def __init__(
        self,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        dims=None,
        batch_size: int = 32,
        major_voting: bool = False,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            dims=dims,
        )
        self.batch_size = batch_size
        self.major_voting = major_voting
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        """Called before fit (train + validate), validate, test, or predict.

        Args:
            stage(str, optional): either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        if self.major_voting:
            self.compute_major_votes()
        
        print(self.data, self.annotations)

        data = self.data
        annotations = self.annotations
        
        self.text_id_idx_dict = (
            data.loc[:, ["text_id"]]
                .reset_index()
                .set_index("text_id")
                .to_dict()["index"]
        )
        annotator_id_category = annotations["annotator_id"].astype("category")
        self.annotator_id_idx_dict = {
            a_id: idx for idx, a_id in enumerate(annotator_id_category.cat.categories)
        }

    def compute_major_votes(self) -> None:
        """Computes mean votes for every text and replaces
        each annotator with dummy annotator with id = 0"""
        annotations = self.annotations

        major_votes = annotations.groupby("text_id")[self.annotation_column].mean()
        if self.task_type != 'regression':
            major_votes = major_votes.round()

        self.annotations = major_votes.reset_index()
        self.annotations["annotator_id"] = 0

    def train_dataloader(self) -> DataLoader:
        """Returns train dataloader."""
        return self._dataloader(self.train_split_names)

    def val_dataloader(self) -> DataLoader:
        """Returns val dataloader."""
        return self._dataloader(self.val_split_names)

    def test_dataloader(self) -> DataLoader:
        """Returns test dataloader."""
        return self._dataloader(self.test_split_names)

    def whole_dataset_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Returns dataloader containing data from all splits.

        Args:
            shuffle (bool, optional): Flag indicating if dataloader should shuffle the data.
                Defaults to True.
        """
        return self._dataloader(
            splits=self.test_split_names + self.val_split_names + self.test_split_names,
            shuffle=shuffle
        )

    def _dataloader(self, splits: List[str], shuffle: Optional[bool] = None) -> DataLoader:
        """Returns dataloader for given dataset splits.

        Args:
            splits (list of str): Dataset splits for which dataloader is obtained.
            shuffle (bool, optional): Flag indicating if dataloader should shuffle the data.
                Defaults to None.

        Returns:
            DataLoader: Dataloader for a given split.
        """
        dataset = self.get_dataset(splits)

        if shuffle is None:
            shuffle = splits == self.train_split_names

        if shuffle:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.RandomSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False,
            )
        else:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SequentialSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False,
            )

        return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=self.num_workers)

    def get_dataset(self, splits: List[str]) -> BaseDataset:
        """Gets BaseDataset object for given splits.

        Args:
            splits (list of str): Dataset splits for which dataset is obtained.

        Returns:
            BaseDataset: returned dataset object.
        """
        X, y = self._get_data_by_split(splits)
        text_features = self._get_text_features()
        dataset = BaseDataset(X, y, self.task_type, self.task_name,
                              self.task_category, text_features)
        return dataset

    def _get_data_by_split(self, splits: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Gets data for given splits.

        The returned data are in raw format.

        Args:
           splits (list of str): Dataset splits for which data are obtained.

        Returns:
           A tuple (X, y) where X is data with text and annotators ids and y is ground truth data.
        """
        data = self.data

        df = self.annotations.loc[
            self.annotations.text_id.isin(data[data.split.isin(splits)].text_id.values)
        ]
        X = df.loc[:, ["text_id", "annotator_id"]]
        y = df[self.annotation_column]
        print(X, y)

        X["text_id"] = X["text_id"].apply(lambda r_id: self.text_id_idx_dict[r_id])
        X["annotator_id"] = X["annotator_id"].apply(
            lambda w_id: self.annotator_id_idx_dict[w_id]
        )

        X, y = X.values, y.values

        if y.ndim < 2:
            y = y[:, None]

        return X, y

    def _get_text_features(self) -> Dict[str, Any]:
        """Returns dictionary of features of all texts in the dataset.
        Each feature should be a numpy array of whatever dtype, with length equal to number of
        texts in the dataset. Features can be used by models during training.

        It always consists of raw texts. In the case of tasks where second texts are present, they
        are appended. For sequence labelling tasks, lists of tokenized words is added as well.

        Returns:
           A dictionary of text features.
        """
        text_features = {
            "raw_texts": self.data[self.text_column].values,
        }
        if hasattr(self, 'text_2_column') and self.text_2_column is not None:
            text_features['raw_2nd_texts'] = self.data[self.text_2_column].values

        if hasattr(self, 'tokens_column') and self.tokens_column is not None:
            text_features['tokens'] = self.data[self.tokens_column].to_numpy()

        return text_features

    def normalize_labels(self) -> None:
        """Normalizes ground truth numeric data,

        It applies minmax normalization for each ground truth column separately.
        Applicable for regression tasks.
        """
        annotation_column = self.annotation_column
        df = self.annotations

        mins = df.loc[:, annotation_column].values.min(axis=0)
        df.loc[:, annotation_column] = (df.loc[:, annotation_column] - mins)

        maxes = df.loc[:, annotation_column].values.max(axis=0)
        df.loc[:, annotation_column] = df.loc[:, annotation_column] / maxes
