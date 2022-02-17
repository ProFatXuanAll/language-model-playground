"""Setup fixtures for testing :py:mod:`lmp.tknzr`."""

from typing import Dict

import pytest


@pytest.fixture(params=[False, True])
def is_uncased(request) -> bool:
  """Respect cases if set to ``False``."""
  return request.param


@pytest.fixture(params=[-1, 10000])
def max_vocab(request) -> int:
  """Maximum vocabulary size."""
  return request.param


@pytest.fixture(params=[0, 10])
def min_count(request) -> int:
  """Minimum token occurrence counts."""
  return request.param


@pytest.fixture(params=[
  {
    'input': 'ABC',
    'output': 'abc'
  },
  {
    'input': 'abc',
    'output': 'abc'
  },
])
def uncased_txt(request) -> Dict[str, str]:
  """Case-insensitive text."""
  return request.param
