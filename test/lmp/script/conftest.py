"""Setup fixtures for testing :py:mod:`lmp.script.train_model`."""

import argparse
import os
from typing import Callable, List

import pytest

import lmp.util.cfg
import lmp.util.path
import lmp.util.tknzr
from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def cfg_file_path(clean_dir_finalizer_factory: Callable[[str], None], exp_name: str, request) -> str:
  """Mock configuration file path.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, lmp.util.cfg.FILE_NAME)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_file_path


@pytest.fixture
def ckpt_dir_path(clean_dir_finalizer_factory: Callable[[str], None], exp_name: str, request) -> str:
  """Directory containing model checkpoints.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_dir_path


@pytest.fixture
def eval_log_dir_path(clean_dir_finalizer_factory: Callable[[str], None], model_exp_name: str, request) -> str:
  """Directory containing model evaluation logs.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.util.path.LOG_PATH, model_exp_name)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_dir_path


@pytest.fixture
def model(tknzr: BaseTknzr) -> BaseModel:
  """:py:class:`lmp.model.BaseModel` instance."""
  return ElmanNet(d_emb=2, d_hid=2, p_emb=0.1, p_hid=0.1, tknzr=tknzr)


@pytest.fixture
def model_exp_name(
  ckpts: List[int],
  clean_dir_finalizer_factory: Callable[[str], None],
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
    args=argparse.Namespace(exp_name=exp_name, tknzr_exp_name=tknzr_exp_name, max_seq_len=max_seq_len),
    exp_name=exp_name
  )

  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return exp_name


@pytest.fixture
def tknzr() -> BaseTknzr:
  """:py:class:`lmp.tknzr.BaseTknzr` instance."""
  tknzr = CharTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  return tknzr


@pytest.fixture
def tknzr_exp_name(
  clean_dir_finalizer_factory: Callable[[str], None],
  exp_name: str,
  tknzr: BaseTknzr,
  request,
) -> str:
  """Tokenizer experiment name."""
  exp_name = f'{exp_name}-tokenizer'
  lmp.util.tknzr.save(exp_name=exp_name, tknzr=tknzr)

  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return exp_name


@pytest.fixture
def tknzr_file_path(clean_dir_finalizer_factory: Callable[[str], None], exp_name: str, request) -> str:
  """Tokenizer pickle file path.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, lmp.util.tknzr.FILE_NAME)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_file_path


@pytest.fixture
def train_log_dir_path(clean_dir_finalizer_factory: Callable[[str], None], exp_name: str, request) -> str:
  """Directory containing model training logs.

  After testing, clean up files and directories created during test.
  """
  abs_dir_path = os.path.join(lmp.util.path.LOG_PATH, exp_name)
  request.addfinalizer(clean_dir_finalizer_factory(abs_dir_path))
  return abs_dir_path
