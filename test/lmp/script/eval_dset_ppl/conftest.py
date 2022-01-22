"""Setup fixtures for testing :py:mod:`lmp.script.eval_dset_ppl`."""

import argparse
import os
from typing import List

import pytest

import lmp.util.cfg
import lmp.util.model
import lmp.util.path
from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def batch_size() -> int:
  """Batch size."""
  return 128


@pytest.fixture
def ckpts() -> List[int]:
  return [0, 1, 2]


@pytest.fixture
def max_seq_len() -> int:
  """Maximum sequence length."""
  return 128


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
def model(tknzr: BaseTknzr) -> BaseModel:
  """:py:class:`lmp.model.BaseModel` instance."""
  return ElmanNet(d_emb=10, tknzr=tknzr)


@pytest.fixture
def model_exp_name(
  ckpts: List[int],
  exp_name: str,
  max_seq_len: int,
  model: BaseModel,
  request,
  tknzr_exp_name: str,
) -> str:
  """Language model experiment name."""
  exp_name = f'{exp_name}-model'
  for ckpt in ckpts:
    lmp.util.model.save(ckpt=ckpt, exp_name=exp_name, model=model)
  lmp.util.cfg.save(
    args=argparse.Namespace(
      exp_name=exp_name,
      tknzr_exp_name=tknzr_exp_name,
      max_seq_len=max_seq_len,
    ),
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


@pytest.fixture
def log_dir_path(model_exp_name: str, request) -> str:
  """Directory containing language model evaluation logs."""
  abs_dir_path = os.path.join(lmp.util.path.LOG_PATH, model_exp_name)

  def fin() -> None:
    for file_name in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, file_name))
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_dir_path
