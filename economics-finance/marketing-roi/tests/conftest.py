import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def cfg():
    from roikit import config
    return config.load()


@pytest.fixture(scope="session")
def result(cfg):
    from roikit import pipeline
    return pipeline.run(cfg)
