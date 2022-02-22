"""Setup fixtures for testing :py:mod:`lmp.model`."""

import pytest

from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def tknzr() -> BaseTknzr:
  """:py:class:`lmp.tknzr.BaseTknzr` instance."""
  tknzr = CharTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  return tknzr


@pytest.fixture
def tknzr_exp_name(exp_name: str) -> str:
  """Tokenizer experiment name."""
  return f'{exp_name}-tokenizer'
