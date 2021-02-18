r"""Setup fixture for testing :py:mod:`lmp.tknzr`."""

import pytest


@pytest.fixture(params=[])
def max_seq_len(request) -> str:
    r"""Infer instance attribute ``max_seq_len``."""
    return request.param
