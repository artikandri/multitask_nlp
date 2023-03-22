from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from typing import Dict, List, Type

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.datasets.multitask_dataset import MultiTaskDataset, RoundRobinMTDataset


class MultiTaskDataModule(LightningDataModule):
    """Datamodule used for processing in multitask manner.

    It takes datamodules for single tasks and provides dataloaders with batches which homogeneous
    for some task.

    Attributes:
        tasks_datamodules (list of BaseDataModule): Datamodules for each task.
        multitask_dataset_cls (type of MultiTaskDataset): Class of multitask dataset used by train
            dataloader. Different classes of multitask dataset use different methods to sample task
            batches during training stage.
        multitask_dataset_args (dict, optional): Additional keyword arguments for train multitask
            dataset. Defaults to None.
        num_workers (int, optional): Number of workers used by dataloaders. Defaults to 0.
    """
    def __init__(
        self,
        tasks_datamodules: List[BaseDataModule],
        multitask_dataset_cls: Type[MultiTaskDataset],
        multitask_dataset_args: Dict = None,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        dims=None,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            dims=dims,
        )
        self.tasks_datamodules = tasks_datamodules
        self.multitask_dataset_cls = multitask_dataset_cls
        if multitask_dataset_args is None:
            multitask_dataset_args = {}

        self.multitask_dataset_args = multitask_dataset_args
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Returns train dataloader.

        Returned dataloader consists of batches coming from dataloaders for each of separate tasks.
        Hence, each batch yielded by the train dataloader comprises data from only one task.

        Task sampling method for training dataloader is dictated by class of multitask dataset.
        """
        task_dataloaders = [datamodule.train_dataloader() for datamodule in self.tasks_datamodules]
        dataset = self.multitask_dataset_cls(task_dataloaders, **self.multitask_dataset_args)
        return self._dataloader(dataset=dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns val dataloader.

        Tasks are sampled in round-robin manner so all validation records for each task will
        be sampled.
        """
        task_dataloaders = [datamodule.val_dataloader() for datamodule in self.tasks_datamodules]
        dataset = RoundRobinMTDataset(task_dataloaders)
        return self._dataloader(dataset=dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns test dataloader.

        Tasks sampling the same as for validation dataloader.
        """
        task_dataloaders = [datamodule.test_dataloader() for datamodule in self.tasks_datamodules]
        dataset = RoundRobinMTDataset(task_dataloaders)
        return self._dataloader(dataset=dataset, shuffle=False)
    
    def get_dataset(self, splits: List[str]) -> RoundRobinMTDataset:
        task_dataloaders = [datamodule.test_dataloader() for datamodule in self.tasks_datamodules]
        dataset = RoundRobinMTDataset(task_dataloaders)
        return dataset

    def _dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Returns dataloader for a given dataset object.

        batch_size is set to None so yielded data are not batched. It is done in the such way
        because data are already batched by dataloaders for single tasks.

        Args:
            dataset (Dataset): Dataset object for which dataloader is created.
            shuffle (bool): Flag indicating if dataloader should shuffle the data.

        Returns:
            Dataloader for a passed dataset.
        """
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            shuffle=shuffle
        )

