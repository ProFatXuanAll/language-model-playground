r"""Setup fixtures for testing :py:mod:`lmp`."""

import uuid
from typing import Dict

import pytest


@pytest.fixture
def exp_name() -> str:
    r"""Test experiment name.

    Experiment name is used to save experiment result, such as tokenizer
    configuration, model checkpoint and logging.

    Returns
    =======
    str
        Experiment name with the format ``test-uuid``.
    """
    return 'test-' + str(uuid.uuid4())


@pytest.fixture(params=[
    {'input': '０', 'output': '0'},  # Full-width to half-width.
    {'input': 'é', 'output': 'é'},  # NFKD to NFKC.
])
def non_nfkc_txt(request) -> Dict[str, str]:
    r"""Text with Non-NFKC normalized characters."""
    return request.param


@pytest.fixture(params=[
    {'input': 'a  b  c', 'output': 'a b c'},
    {'input': '  ', 'output': ''},
])
def cws_txt(request) -> Dict[str, str]:
    r"""Text with consecutive whitespaces."""
    return request.param


@pytest.fixture(params=[
    {'input': ' abc', 'output': 'abc'},
    {'input': 'abc ', 'output': 'abc'},
    {'input': ' abc ', 'output': 'abc'},
])
def htws_txt(request) -> Dict[str, str]:
    r"""Text with whitespaces at head and tail."""
    return request.param
