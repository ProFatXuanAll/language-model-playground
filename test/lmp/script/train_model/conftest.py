"""Setup fixtures for testing :py:mod:`lmp.script.train_model`."""

import os

import pytest

import lmp.util.cfg
import lmp.util.path
import lmp.util.tknzr
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
def cfg_file_path(exp_name: str, request) -> str:
  """Clean up saved configuration file."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, lmp.util.cfg.FILE_NAME)

  def fin() -> None:
    if os.path.exists(abs_file_path):
      os.remove(abs_file_path)
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_file_path


@pytest.fixture
def ckpt_dir_path(exp_name: str, request) -> str:
  """Directory containing model checkpoints.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)

  def fin() -> None:
    for file_name in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, file_name))
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_dir_path


@pytest.fixture
def ckpt_step() -> int:
  """Checkpoint step."""
  return 10


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


@pytest.fixture
def eps() -> float:
  """Epsilon."""
  return 1e-8


@pytest.fixture
def log_dir_path(exp_name: str, request) -> str:
  """Directory containing model training logs.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.util.path.LOG_PATH, exp_name)

  def fin() -> None:
    for file_name in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, file_name))
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_dir_path


@pytest.fixture
def log_step() -> int:
  """Log step."""
  return 10


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
  return 4


@pytest.fixture(params=[2, 4])
def n_blk(request) -> int:
  """Number of memory cell blocks."""
  return request.param


@pytest.fixture
def n_epoch() -> int:
  """Number of training epochs."""
  return 2


@pytest.fixture
def p_emb(request) -> float:
  """Embedding dropout probability."""
  return 0.1


@pytest.fixture
def p_hid(request) -> float:
  """Hidden units dropout probability."""
  return 0.1


@pytest.fixture
def seed() -> int:
  """Random seed."""
  return 42


@pytest.fixture
def tknzr() -> BaseTknzr:
  """:py:class:`lmp.tknzr.BaseTknzr` instance."""
  tknzr = CharTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  return tknzr


@pytest.fixture
def tknzr_exp_name(exp_name: str, request, tknzr: BaseTknzr) -> str:
  """Tokenizer experiment name."""
  exp_name = f'{exp_name}-tokenizer'
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)

  def fin() -> None:
    for file_name in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, file_name))
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return exp_name


@pytest.fixture
def warmup_step() -> int:
  """Warmup step."""
  return 1000


@pytest.fixture
def wd() -> float:
  """Weight decay coefficient of AdamW."""
  return 1e-2
