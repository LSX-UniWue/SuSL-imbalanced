from lightning import Trainer
from torch import float32
from torch.nn import Sequential, ReLU, Identity, Flatten, Upsample, Linear, Conv2d
from torch.utils.data import RandomSampler
from torchmetrics import MetricCollection
from torchvision.transforms.v2 import PILToTensor, Compose, ToDtype, Lambda

from datasets.imbalanced_mnist import ImbalancedMNIST
from implementations.data_module import SemiUnsupervisedDataModule
from implementations.priors import MixturePrior
from implementations.sampler import MeanSampler
from susl_base.data.susl_dataset import LabeledDatasetFacade
from susl_base.data.utils import create_susl_dataset
from susl_base.metrics.cluster_and_label import ClusterAccuracy
from susl_base.networks.gmm_dgm import EntropyRegularizedGaussianMixtureDeepGenerativeModel
from susl_base.networks.latent_layer import LatentLayer
from susl_base.networks.lightning import LightningGMMModel
from susl_base.networks.losses import EntropyGaussianMixtureDeepGenerativeLoss
from susl_base.networks.misc import Reshape
from susl_base.networks.variational_layer import BernoulliVariationalLayer, GaussianVariationalLayer


def run_cnn() -> None:
    # Create datasets
    transforms = Compose(
        [
            PILToTensor(),
            ToDtype(float32, scale=True),
            Lambda(lambda x: (x >= 0.5).float()),
        ]
    )

    train_dataset = ImbalancedMNIST(mode="train", transform=transforms)
    train_dataset_labeled, train_dataset_unlabeled, class_mapper = create_susl_dataset(
        dataset=train_dataset, num_labels=0.2, classes_to_hide=[5, 6, 7, 8, 9]
    )
    train_dataset_labeled_class_map = [train_dataset_labeled[i]["y_l"] for i in range(len(train_dataset_labeled))]
    validation_dataset = ImbalancedMNIST(mode="val", transform=transforms)
    test_dataset = ImbalancedMNIST(mode="test", transform=transforms)

    # Create model
    n_l, n_aug, n_classes = 5, 40, 10
    n_x, n_y, n_z = 28 * 28, n_l + n_aug, 50
    datamodule = SemiUnsupervisedDataModule(
        train_dataset_labeled=train_dataset_labeled,
        train_dataset_unlabeled=train_dataset_unlabeled,
        validation_dataset=LabeledDatasetFacade(
            validation_dataset, indices=list(range(len(validation_dataset))), class_mapper=class_mapper
        ),
        test_dataset=LabeledDatasetFacade(
            test_dataset, indices=list(range(len(test_dataset))), class_mapper=class_mapper
        ),
        sampler_labelled=MeanSampler(
            class_map=train_dataset_labeled_class_map, num_samples=len(train_dataset_unlabeled)
        ),
        sampler_unlabelled=RandomSampler(data_source=train_dataset_unlabeled),
        batch_size=128,
    )

    q_y_x_module = Sequential(
        Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2),
        ReLU(),
        Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
        ReLU(),
        Flatten(),
        Linear(in_features=64 * 7 * 7, out_features=n_y),
    )
    p_x_z_module = BernoulliVariationalLayer(
        feature_extractor=Sequential(
            Linear(in_features=n_z, out_features=64 * 7 * 7),
            Reshape((-1, 64, 7, 7)),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Upsample(scale_factor=2),
            Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1),
            ReLU(),
            Upsample(scale_factor=2),
        ),
        module_init=Conv2d,
        out_channels=1,
        in_channels=1,
        kernel_size=1,
    )
    p_z_y_module = GaussianVariationalLayer(feature_extractor=Identity(), in_features=n_y, out_features=n_z)
    q_z_xy_module = GaussianVariationalLayer(
        feature_extractor=LatentLayer(
            pre_module=Sequential(
                Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2),
                ReLU(),
                Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
                ReLU(),
                Flatten(),
            ),
            post_module=Sequential(Linear(in_features=64 * 7 * 7 + n_y, out_features=128), ReLU()),
        ),
        out_features=n_z,
        in_features=128,
    )
    log_priors = MixturePrior().get_prior(n_l=n_l, n_aug=n_aug).log()
    model = EntropyRegularizedGaussianMixtureDeepGenerativeModel(
        n_y=n_y,
        n_z=n_z,
        n_x=n_x,
        q_y_x_module=q_y_x_module,
        p_x_z_module=p_x_z_module,
        p_z_y_module=p_z_y_module,
        q_z_xy_module=q_z_xy_module,
        log_priors=log_priors,
    )
    print(model)
    # Create trainer and run
    lt_model = LightningGMMModel(
        model=model,
        loss_fn=EntropyGaussianMixtureDeepGenerativeLoss(),
        val_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_classes, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_classes, average="macro"),
            },
            prefix="val_",
        ),
        test_metrics=MetricCollection(
            metrics={
                "micro_accuracy": ClusterAccuracy(num_classes=n_classes, average="micro"),
                "macro_accuracy": ClusterAccuracy(num_classes=n_classes, average="macro"),
            },
            prefix="test_",
        ),
    )
    trainer = Trainer(max_epochs=10, check_val_every_n_epoch=2)
    trainer.fit(model=lt_model, datamodule=datamodule)
    trainer.test(model=lt_model, datamodule=datamodule)


if __name__ == "__main__":
    run_cnn()
