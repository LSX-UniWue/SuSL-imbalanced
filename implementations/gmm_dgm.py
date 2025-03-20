from typing import Callable, Dict

from torch import Tensor, no_grad, max, rand_like
from torch.nn import Module
from torch.nn.functional import cross_entropy

from susl_base.networks.gmm_dgm import GaussianMixtureDeepGenerativeModel


class BiSamplingModel(Module):
    def __init__(
        self,
        model: GaussianMixtureDeepGenerativeModel,
        sampler_a: Tensor,
        sampler_b: Tensor,
        alpha_function: Callable[[int], float],
        label_threshold: float = 0.95,
        sampler_a_augmented: Tensor = None,
        sampler_b_augmented: Tensor = None,
    ) -> None:
        super().__init__()
        self.__model = model
        self.__sampler_a = sampler_a
        self.__sampler_b = sampler_b
        self.__alpha_function = alpha_function
        self.__sampler_a_augmented = sampler_a_augmented
        self.__sampler_b_augmented = sampler_b_augmented
        self.__label_threshold = label_threshold
        self.__current_epoch = 0

    # Adapted from https://github.com/TACJu/Bi-Sampling/blob/main/fix_BiS.py
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Get weak labels
        with no_grad():
            weak_labels = self.__model.predict(data["x_u"])

        # Calculate helpers
        alpha = self.__alpha_function(self.__current_epoch)
        mixture_sampler = alpha * self.__sampler_a + (1 - alpha) * self.__sampler_b
        max_p, p_hat = max(weak_labels, dim=1)

        # Create labelled selection mask
        select_mask1 = max_p >= self.__label_threshold
        select_mask2 = rand_like(weak_labels) < mixture_sampler[p_hat]
        select_mask3 = weak_labels < mixture_sampler.nelement()
        mask = select_mask1 & select_mask2 & select_mask3

        # Get real predictions
        labels = self.__model.predict(data["x_u"])
        losses = self.__model(data)

        # Update losses
        losses["reg_labelled"] = losses["reg_labelled"] + cross_entropy(input=labels, target=p_hat)[mask].mean()

        # Create unknown labelled selection mask
        if self.__sampler_a_augmented is not None and self.__sampler_b_augmented is not None:
            select_mask1 = max_p >= self.__label_threshold
            select_mask2 = rand_like(weak_labels) < mixture_sampler[p_hat]
            select_mask3 = weak_labels >= mixture_sampler.nelement()
            mask = select_mask1 & select_mask2 & select_mask3
            losses["reg_unlabelled"] = losses["reg_unlabelled"] + cross_entropy(input=labels, target=p_hat)[mask].mean()

        return losses
