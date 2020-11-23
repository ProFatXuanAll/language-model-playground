r"""Setup fixture for testing :py:mod:`lmp`."""

import uuid

import pytest


@pytest.fixture
def exp_name() -> str:
    r"""Experiment name for test.

    Experiment name is used to save experiment result, such as tokenizer
    configuration, model checkpoint and logging.

    Returns
    =======
    str
        Experiment name with the format ``test-uuid``.
    """
    return 'test-' + str(uuid.uuid4())
