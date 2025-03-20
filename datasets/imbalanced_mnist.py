from typing import Literal

from numpy import load
from torch import from_numpy
from torchvision.datasets import MNIST


class ImbalancedMNIST(MNIST):
    resources = "./datasets/imbalanced_imbalanced.npz"
    mode = "train"

    def __init__(
        self,
        mode: Literal["train", "val", "test"] = "train",
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root="/tmp", transform=transform, target_transform=target_transform)
        self.mode = mode

    def download(self) -> None:
        pass

    def _check_exists(self) -> bool:
        return True

    def _load_data(self):
        with load(self.resources) as file:
            data = from_numpy(file[f"x_{self.mode}"]).reshape(-1, 28, 28)
            targets = from_numpy(file[f"y_{self.mode}"])

        return data, targets
