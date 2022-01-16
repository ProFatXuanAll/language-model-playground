"""Setup fixtures for testing :py:mod:`lmp`."""

import uuid
from typing import Dict

import pytest


@pytest.fixture
def exp_name() -> str:
  """Test experiment name.

  Experiment name is used to save experiment result, such as tokenizer
  configuration, model checkpoint and logging.

  Returns
  -------
  str
      Experiment name with the format ``test-uuid``.
  """
  return 'test-' + str(uuid.uuid4())


@pytest.fixture(
  params=[
    # Full-width to half-width.
    {
      'input': '０',
      'output': '0'
    },
    # NFKD to NFKC.
    {
      'input': 'é',
      'output': 'é'
    },
  ]
)
def nfkc_txt(request) -> Dict[str, str]:
  """Normalize text with NFKC."""
  return request.param


@pytest.fixture(params=[
  {
    'input': 'a  b  c',
    'output': 'a b c'
  },
  {
    'input': '  ',
    'output': ''
  },
])
def ws_collapse_txt(request) -> Dict[str, str]:
  """Collapse consecutive whitespaces."""
  return request.param


@pytest.fixture(
  params=[
    {
      'input': ' abc',
      'output': 'abc'
    },
    {
      'input': 'abc ',
      'output': 'abc'
    },
    {
      'input': ' abc ',
      'output': 'abc'
    },
  ]
)
def ws_strip_txt(request) -> Dict[str, str]:
  """Strip whitespaces at head and tail."""
  return request.param
