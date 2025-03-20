from typing import Literal, List, Iterator, Callable

from numpy.random import choice
from torch.utils.data import Sampler, BatchSampler


def get_decay_function(decay_strategy: str, max_epochs: int) -> Callable[[int], float]:
    match decay_strategy:
        case "equal":
            return lambda _t: 0.5
        case "linear":
            return lambda _t: 1 - _t / max_epochs
        case "cosine":
            from math import cos, pi

            return lambda _t: cos((_t / max_epochs) * (pi / 2))
        case "parabolic":
            return lambda _t: 1 - (_t / max_epochs) ** 2


class BiSampler(Sampler[List[int]]):
    def __init__(
        self,
        sampler_a: Sampler,
        sampler_b: Sampler,
        max_epochs: int,
        batch_size: int,
        drop_last: bool,
        decay_strategy: Literal["equal", "linear", "cosine", "parabolic"] = "parabolic",
    ) -> None:
        super().__init__()
        self.__sampler_a = BatchSampler(sampler_a, batch_size=batch_size, drop_last=drop_last)
        self.__sampler_b = BatchSampler(sampler_b, batch_size=batch_size, drop_last=drop_last)
        self.__alpha_func = get_decay_function(decay_strategy=decay_strategy, max_epochs=max_epochs)
        self.__current_epoch = -1
        self.__iter_sampler_a, self.__iter_sampler_b = None, None
        self.on_epoch_end()

    def __len__(self) -> int:
        return len(self.__sampler_a)

    def __iter__(self) -> Iterator[int]:
        try:
            batch_a = next(self.__iter_sampler_a)
            batch_b = next(self.__iter_sampler_b)
        except StopIteration:
            self.on_epoch_end()
            batch_a = next(self.__iter_sampler_a)
            batch_b = next(self.__iter_sampler_b)
        bs = len(batch_a)
        alpha = self.__alpha_func(self.__current_epoch)
        num_a = int(alpha * bs)
        # Yield sampler one
        yield from choice(batch_a, size=num_a)
        # Yield sampler two
        yield from choice(batch_b, size=bs - num_a)

    def on_epoch_end(self) -> None:
        self.__current_epoch += 1
        self.__iter_sampler_a = iter(self.__sampler_a)
        self.__iter_sampler_b = iter(self.__sampler_b)
