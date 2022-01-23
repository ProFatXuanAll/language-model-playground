"""Setup fixtures for testing :py:mod:`lmp.util.infer`."""

import pytest


@pytest.fixture
def max_seq_len() -> int:
  """Maximum sequence length."""
  return 128
