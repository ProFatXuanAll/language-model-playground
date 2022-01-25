"""Setup fixtures for testing :py:mod:`lmp.script.train_tknzr`."""

import argparse
import os

import pytest

import lmp.util.cfg
import lmp.util.path
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
  abs_file_path = os.path.join(abs_dir_path, lmp.util.cfg.CFG_NAME)

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
def d_cell(request) -> int:
  """Memory cell dimension."""
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
  return 128


@pytest.fixture(params=[2, 4])
def n_cell(request) -> int:
  """Number of memory cells."""
  return request.param


@pytest.fixture
def n_epoch() -> int:
  """Number of training epochs."""
  return 2


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
def tknzr_exp_name(exp_name: str, request, tknzr: BaseTknzr) -> str:
  """Tokenizer experiment name."""
  exp_name = f'{exp_name}-tokenizer'
  tknzr.save(exp_name=exp_name)
  lmp.util.cfg.save(args=argparse.Namespace(exp_name=exp_name, tknzr_name=tknzr.tknzr_name), exp_name=exp_name)
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)

  def fin() -> None:
    for file_name in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, file_name))
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return exp_name


@pytest.fixture
def wd() -> float:
  """Weight decay coefficient of AdamW."""
  return 1e-2
