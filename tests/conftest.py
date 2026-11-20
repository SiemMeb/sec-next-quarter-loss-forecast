import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))


@pytest.fixture(scope="session")
def panel():
    from synthetic import make_panel
    from lossforecast.data import normalise
    return normalise(make_panel())
