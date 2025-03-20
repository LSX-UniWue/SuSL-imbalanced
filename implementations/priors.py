from abc import ABC, abstractmethod

from torch import Tensor, tensor, arange, hstack, full
from torch.distributions import Normal


class PriorGenerator(ABC):
    @abstractmethod
    def get_singleton_prior(self, n: int) -> Tensor:
        pass

    def get_prior(self, n_l: int, n_aug: int) -> Tensor:
        # Unsupervised
        if n_l <= 0:
            return self.get_singleton_prior(n=n_aug)
        # (Semi-)Supervised
        if n_aug <= 0:
            return self.get_singleton_prior(n=n_l)
        # SuSL
        labelled_prior = self.get_singleton_prior(n=n_l)
        unlabelled_prior = self.get_singleton_prior(n=n_aug)
        return 0.5 * hstack((labelled_prior, unlabelled_prior))


class UniformPrior(PriorGenerator):
    def get_singleton_prior(self, n: int) -> Tensor:
        return full((n,), fill_value=1 / n)


class NormalPrior(PriorGenerator):
    def get_singleton_prior(self, n: int) -> Tensor:
        dist = Normal(loc=0, scale=n**0.5)
        prior = dist.cdf(arange(start=1, end=n + 1)) - dist.cdf(arange(start=0, end=n))
        norm = dist.cdf(tensor(n)) - dist.cdf(tensor(0))
        return prior / norm


class MixturePrior(PriorGenerator):
    def __init__(
        self, labeled_generator: PriorGenerator = UniformPrior(), unlabeled_generator: PriorGenerator = NormalPrior()
    ) -> None:
        super().__init__()
        self.__labeled_generator = labeled_generator
        self.__unlabeled_generator = unlabeled_generator

    # Will not be called, for consistency only
    def get_singleton_prior(self, n: int) -> Tensor:
        raise NotImplementedError("Invalid method for MixturePrior.")

    def get_prior(self, n_l: int, n_aug: int) -> Tensor:
        # Unsupervised
        if n_l <= 0:
            return self.__unlabeled_generator.get_singleton_prior(n=n_aug)
        # (Semi-)Supervised
        if n_aug <= 0:
            return self.__labeled_generator.get_singleton_prior(n=n_l)
        # SuSL
        labelled_prior = self.__labeled_generator.get_singleton_prior(n=n_l)
        unlabelled_prior = self.__unlabeled_generator.get_singleton_prior(n=n_aug)
        return 0.5 * hstack((labelled_prior, unlabelled_prior))


if __name__ == "__main__":
    for generator in (UniformPrior(), NormalPrior(), MixturePrior()):
        print(generator.__class__.__name__)
        # SuSL
        print(generator.get_prior(5, 10))
        print(generator.get_prior(5, 10).sum())
        # UL
        print(generator.get_prior(0, 15))
        print(generator.get_prior(0, 15).sum())
        # SL
        print(generator.get_prior(15, 0))
        print(generator.get_prior(15, 0).sum())
