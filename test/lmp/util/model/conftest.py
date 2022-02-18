"""Setup fixtures for testing :py:mod:`lmp.util.model`."""

import os

import pytest

import lmp.util.path
from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture(params=[1, 2])
def d_blk(request) -> int:
  """Dimension of each memory cell block."""
  return request.param


@pytest.fixture(params=[1, 2])
def d_emb(request) -> int:
  """Embedding dimension."""
  return request.param


@pytest.fixture(params=[1, 2])
def d_hid(request) -> int:
  """Hidden dimension."""
  return request.param


@pytest.fixture(params=[False, True])
def is_uncased(request) -> bool:
  """Respect cases if set to ``False``."""
  return request.param


@pytest.fixture(params=[-1, 10])
def max_vocab(request) -> int:
  """Maximum vocabulary size."""
  return request.param


@pytest.fixture(params=[0, 10])
def min_count(request) -> int:
  """Minimum token occurrence counts."""
  return request.param


@pytest.fixture(params=[1, 2])
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
def tknzr(is_uncased: bool, max_vocab: int, min_count: int) -> BaseTknzr:
  """Tokenizer example."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  return tknzr


@pytest.fixture
def model(d_emb: int, d_hid: int, p_emb: float, p_hid: float, tknzr: BaseTknzr) -> BaseModel:
  """Save model example."""
  return ElmanNet(d_emb=d_emb, d_hid=d_hid, p_emb=p_emb, p_hid=p_hid, tknzr=tknzr)


@pytest.fixture
def ckpt_dir_path(exp_name: str, request) -> str:
  """Clean up saving model checkpoints."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)

  def fin() -> None:
    for file_name in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, file_name))

    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_dir_path
