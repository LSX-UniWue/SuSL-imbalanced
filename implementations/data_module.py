from typing import Dict

from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, Sampler, DataLoader, RandomSampler


class MixedDataset(Dataset):
    def __init__(
        self,
        dataset_labelled: Dataset,
        dataset_unlabelled: Dataset,
        sampler_labelled: Sampler,
    ) -> None:
        super().__init__()
        self.__size = max(len(dataset_labelled), len(dataset_unlabelled))
        self.__dataset_labelled = dataset_labelled
        self.__dataset_unlabelled = dataset_unlabelled
        self.__iterator_labelled, self.__iterator_unlabelled = None, None
        self.__sampler_labelled = sampler_labelled
        self.__sampler_unlabelled = RandomSampler(data_source=dataset_unlabelled)
        self.__reset_iterators()

    def __reset_iterators(self) -> None:
        self.__iterator_labelled = iter(self.__sampler_labelled)
        self.__iterator_unlabelled = iter(self.__sampler_unlabelled)

    def __len__(self) -> int:
        return self.__size

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        try:
            labelled_id = next(self.__iterator_labelled)
            unlabelled_id = next(self.__iterator_unlabelled)
        except StopIteration:
            self.__reset_iterators()
            labelled_id = next(self.__iterator_labelled)
            unlabelled_id = next(self.__iterator_unlabelled)
        return self.__dataset_labelled[labelled_id] | self.__dataset_unlabelled[unlabelled_id]


class SemiUnsupervisedDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_unlabeled: Dataset,
        train_dataset_labeled: Dataset,
        sampler_labelled: Sampler,
        validation_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 32,
    ):
        super().__init__()
        self.__batch_size = batch_size
        self.__train_dataset_unlabeled = train_dataset_unlabeled
        self.__train_dataset_labeled = train_dataset_labeled
        self.__validation_dataset = validation_dataset
        self.__test_dataset = test_dataset
        self.__sampler_labelled = sampler_labelled

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            MixedDataset(
                dataset_labelled=self.__train_dataset_labeled,
                dataset_unlabelled=self.__train_dataset_unlabeled,
                sampler_labelled=self.__sampler_labelled,
            ),
            batch_size=self.__batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.__validation_dataset, batch_size=self.__batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.__test_dataset, batch_size=self.__batch_size)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
