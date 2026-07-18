import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from lss.utils.misc import load_config


def test_load_config():
    config = load_config("lss/config/nuscenes_mini_config.yaml")
    assert config is not None
