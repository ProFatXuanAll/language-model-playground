"""Setup fixtures for testing :py:mod:`lmp.model`."""

import pytest

from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture(params=[2, 4])
def d_blk(request) -> int:
  """Dimension of each memory cell block."""
  return request.param


@pytest.fixture(params=[2, 4])
def d_emb(request) -> int:
  """Embedding dimension."""
  return request.param


@pytest.fixture(params=[2, 4])
def d_hid(request) -> int:
  """Hidden dimension."""
  return request.param


@pytest.fixture(params=[2, 4])
def n_blk(request) -> int:
  """Number of memory cell blocks."""
  return request.param


@pytest.fixture(params=[0.1, 0.5])
def p_emb(request) -> float:
  """Embedding dropout probability."""
  return request.param


@pytest.fixture(params=[0.1, 0.5])
def p_hid(request) -> float:
  """Hidden units dropout probability."""
  return request.param


@pytest.fixture
def tknzr() -> BaseTknzr:
  """:py:class:`lmp.tknzr.BaseTknzr` instance."""
  tknzr = CharTknzr(is_uncased=True, max_seq_len=8, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  return tknzr


@pytest.fixture
def tknzr_exp_name(exp_name: str) -> str:
  """Tokenizer experiment name."""
  return f'{exp_name}-tokenizer'
