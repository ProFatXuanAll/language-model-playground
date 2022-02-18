"""Setup fixtures for testing :py:mod:`lmp.script.gen_txt`."""

import argparse
import os

import pytest

import lmp.util.cfg
import lmp.util.model
import lmp.util.path
import lmp.util.tknzr
from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def ckpt() -> int:
  """Checkpoint number."""
  return 0


@pytest.fixture
def max_seq_len() -> int:
  """Maximum sequence length."""
  return 8


@pytest.fixture
def seed() -> int:
  """Maximum sequence length."""
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
def model(tknzr: BaseTknzr) -> BaseModel:
  """:py:class:`lmp.model.BaseModel` instance."""
  return ElmanNet(d_emb=2, d_hid=2, p_emb=0.1, p_hid=0.1, tknzr=tknzr)


@pytest.fixture
def model_exp_name(ckpt: int, exp_name: str, max_seq_len: int, model: BaseModel, request, tknzr_exp_name: str) -> str:
  """Language model experiment name."""
  exp_name = f'{exp_name}-model'
  lmp.util.model.save(ckpt=ckpt, exp_name=exp_name, model=model)
  lmp.util.cfg.save(
    args=argparse.Namespace(exp_name=exp_name, tknzr_exp_name=tknzr_exp_name, max_seq_len=max_seq_len),
    exp_name=exp_name
  )
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)

  def fin() -> None:
    for file_name in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, file_name))
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return exp_name
