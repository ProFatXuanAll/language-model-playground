"""Setup fixtures for testing :py:mod:`lmp.util.optim`."""

import pytest

from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def beta1() -> float:
  """Beta1 example."""
  return 0.9


@pytest.fixture
def beta2() -> float:
  """Beta2 example."""
  return 0.999


@pytest.fixture
def eps() -> float:
  """eps example."""
  return 1e-8


@pytest.fixture(params=[0.1, 1e-3, 1e-5])
def lr(request) -> float:
  """Beta1 example."""
  return request.param


@pytest.fixture
def tknzr() -> BaseTknzr:
  """Tokenizer example."""
  tknzr = CharTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  return tknzr


@pytest.fixture
def model(tknzr: BaseTknzr) -> BaseModel:
  """Save model example."""
  return ElmanNet(d_emb=2, d_hid=4, p_emb=0.5, p_hid=0.1, tknzr=tknzr)


@pytest.fixture
def warmup_step() -> float:
  """Warm up step example."""
  return 1000


@pytest.fixture
def total_step(warmup_step: int) -> float:
  """Total up step example."""
  return warmup_step + 1000


@pytest.fixture
def wd() -> float:
  """Weight decay example."""
  return 1e-2
