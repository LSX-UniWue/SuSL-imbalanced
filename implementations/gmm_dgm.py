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
        select_mask_labelled = p_hat < mixture_sampler.nelement()
        max_p_labelled, p_hat_labelled = max_p[select_mask_labelled], p_hat[select_mask_labelled]
        select_mask1 = max_p_labelled >= self.__label_threshold
        select_mask2 = rand_like(max_p_labelled) < mixture_sampler[p_hat_labelled]
        mask = select_mask1 & select_mask2

        # Get real predictions
        labels = self.__model.predict(data["x_u"])
        losses = self.__model(data)

        # Update losses
        losses["reg_labelled"] = (
            losses["reg_labelled"]
            + cross_entropy(input=labels[select_mask_labelled], target=p_hat_labelled)[mask].mean()
        )

        # Create unknown labelled selection mask
        if self.__sampler_a_augmented is not None and self.__sampler_b_augmented is not None:
            mixture_sampler_unlabelled = alpha * self.__sampler_a_augmented + (1 - alpha) * self.__sampler_b_augmented
            select_mask_unlabelled = p_hat >= mixture_sampler.nelement()
            max_p_unlabelled, p_hat_unlabelled = max_p[select_mask_unlabelled], p_hat[select_mask_unlabelled]
            select_mask1 = max_p_unlabelled >= self.__label_threshold
            select_mask2 = rand_like(max_p_unlabelled) < mixture_sampler_unlabelled[p_hat_unlabelled]
            mask = select_mask1 & select_mask2

            losses["reg_unlabelled"] = (
                losses["reg_unlabelled"]
                + cross_entropy(input=labels[select_mask_unlabelled], target=p_hat)[mask].mean()
            )

        return losses
