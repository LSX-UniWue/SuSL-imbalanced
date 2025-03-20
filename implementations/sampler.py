from collections import Counter
from typing import Sequence

from torch.utils.data import WeightedRandomSampler


def get_sampling_weights(class_map: Sequence[int], exponent: int) -> Sequence[float]:
    class_counts = Counter(class_map)
    return [class_counts[c] ** exponent for c in class_map]


class MeanSampler(WeightedRandomSampler):
    def __init__(self, class_map: Sequence[int], num_samples: int, generator=None):
        super().__init__(
            weights=get_sampling_weights(class_map=class_map, exponent=-1),
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )


class ReverseSampler(WeightedRandomSampler):
    def __init__(self, class_map: Sequence[int], num_samples: int, generator=None):
        super().__init__(
            weights=get_sampling_weights(class_map=class_map, exponent=-2),
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )


if __name__ == "__main__":
    c_map = 100 * [0] + 10 * [1] + 5 * [2] + 1 * [3]
    m_sampler = MeanSampler(class_map=c_map, num_samples=100_000)
    print(Counter(c_map[i] for i in m_sampler.__iter__()))
    r_sampler = ReverseSampler(class_map=c_map, num_samples=100_000)
    print(Counter(c_map[i] for i in r_sampler.__iter__()))
