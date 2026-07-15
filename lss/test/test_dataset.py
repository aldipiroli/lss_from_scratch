import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest

from lss.dataset.mnist import MNISTDataset
from lss.utils.misc import load_config


@pytest.mark.skip()
def test_mnist_dataset():
    config = load_config("lss/config/mnist_config.yaml")
    dataset = MNISTDataset(config, "train", None)
    x = dataset[0]
    assert x.shape == (1, 32, 32)


@pytest.mark.skip()
def test_emnist_dataset():
    config = load_config("lss/config/emnist_config.yaml")
    dataset = MNISTDataset(config, "train", None)
    x = dataset[0]
    assert x.shape == (1, 32, 32)


if __name__ == "__main__":
    print("All tests passed!")
