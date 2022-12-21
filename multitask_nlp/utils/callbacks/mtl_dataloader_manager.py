import pytorch_lightning as pl

from multitask_nlp.datasets.multitask_dataset import MultiTaskDataset


class ValidDatasetResetter(pl.Callback):
    """Callback for resetting MultiTaskDataset.

    Trainer API performs some sanity steps over validation dataloader. Thus, in the case
    MultitaksDataset its iterators have called few next(). To assure that during an actual
    validation, these iterators are at their start positions, they have to be reset.
    """
    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for val_dl in trainer.val_dataloaders:
            val_dataset = val_dl.dataset
            if isinstance(val_dataset, MultiTaskDataset):
                val_dataset.task_iterators = [iter(dl) for dl in val_dataset.task_dataloaders]
