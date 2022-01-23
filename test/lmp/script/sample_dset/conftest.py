"""Setup fixtures for testing :py:mod:`lmp.script.sample_dset`."""

import pytest


@pytest.fixture(params=[0, 1, 2])
def idx(request) -> int:
  """Sample index."""
  return request.param


@pytest.fixture
def seed() -> int:
  """Random seed."""
  return 42
