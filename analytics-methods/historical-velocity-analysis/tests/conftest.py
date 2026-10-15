import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from historykit import synth  # noqa: E402


@pytest.fixture(scope="session")
def book():
    return synth.make_book()


@pytest.fixture(scope="session")
def daily(book):
    return book.daily()
