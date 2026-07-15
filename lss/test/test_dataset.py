import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest

from lss.utils.misc import load_config






if __name__ == "__main__":
    print("All tests passed!")
