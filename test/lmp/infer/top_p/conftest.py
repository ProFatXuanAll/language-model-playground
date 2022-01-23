"""Setup fixtures for testing :py:class:`lmp.infer.TopPInfer.`."""

import pytest


@pytest.fixture
def p() -> float:
  """``p`` in the top-p."""
  return 0.9
