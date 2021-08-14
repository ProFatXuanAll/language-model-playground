r"""Setup fixture for testing :py:mod:`lmp.tknzr.WsTknzr`."""

import pytest

from lmp.tknzr._ws import WsTknzr


@pytest.fixture
def ws_tknzr():
    r"""Simple WsTknzr instance"""

    return WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
    )


@pytest.fixture
def tk2id():
    r"""Simple tk2id"""

    tk2id = {
        '[bos]': 0,
        '[eos]': 1,
        '[pad]': 2,
        '[unk]': 3,
        'a': 4,
        'b': 5,
        'c': 6,
        '1': 7,
        '2': 8,
        '哈': 9,
    }

    return tk2id
