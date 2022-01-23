"""Setup fixtures for testing :py:class:`lmp.infer.TopKInfer.`."""

import pytest


@pytest.fixture
def k() -> int:
  """``k`` in the top-k."""
  return 5
