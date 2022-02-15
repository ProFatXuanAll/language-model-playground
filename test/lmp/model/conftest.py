"""Setup fixtures for testing :py:mod:`lmp.model`."""

import pytest

from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def batch_size() -> int:
  """Batch size."""
  return 32


@pytest.fixture
def beta1() -> float:
  """Beta 1 coefficient of AdamW."""
  return 0.9


@pytest.fixture
def beta2() -> float:
  """Beta 2 coefficient of AdamW."""
  return 0.999


@pytest.fixture
def ckpt_step() -> int:
  """Checkpoint step."""
  return 500


@pytest.fixture(params=[2, 4])
def d_blk(request) -> int:
  """Dimension of each memory cell block."""
  return request.param


@pytest.fixture(params=[10, 20])
def d_emb(request) -> int:
  """Embedding dimension."""
  return request.param


@pytest.fixture
def eps() -> float:
  """Epsilon."""
  return 1e-8


@pytest.fixture
def log_step() -> int:
  """Log step."""
  return 1000


@pytest.fixture
def lr() -> float:
  """Learning rate."""
  return 5e-5


@pytest.fixture
def max_norm() -> float:
  """Gradient clipping max norm."""
  return 1.0


@pytest.fixture
def max_seq_len() -> int:
  """Maximum sequence length."""
  return 128


@pytest.fixture(params=[2, 4])
def n_blk(request) -> int:
  """Number of memory cell blocks."""
  return request.param


@pytest.fixture
def n_epoch() -> int:
  """Number of training epochs."""
  return 10


@pytest.fixture
def seed() -> int:
  """Random seed."""
  return 42


@pytest.fixture
def tknzr() -> BaseTknzr:
  """:py:class:`lmp.tknzr.BaseTknzr` instance."""
  return CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=0,
    tk2id={
      CharTknzr.bos_tk: CharTknzr.bos_tkid,
      CharTknzr.eos_tk: CharTknzr.eos_tkid,
      CharTknzr.pad_tk: CharTknzr.pad_tkid,
      CharTknzr.unk_tk: CharTknzr.unk_tkid,
      'a': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
      'b': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 2,
      'c': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 3,
    },
  )


@pytest.fixture
def tknzr_exp_name(exp_name: str) -> str:
  """Tokenizer experiment name."""
  return f'{exp_name}-tokenizer'


@pytest.fixture
def wd() -> float:
  """Weight decay coefficient of AdamW."""
  return 1e-2
