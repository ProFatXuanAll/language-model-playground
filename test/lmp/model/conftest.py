"""Setup fixtures for testing :py:mod:`lmp.model`."""

import os

import pytest

import lmp.util.path
from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture(params=[
  1,
  2,
])
def d_emb(request) -> int:
  return request.param


@pytest.fixture(params=[
  1,
  2,
])
def d_hid(request) -> int:
  return request.param


@pytest.fixture(params=[
  1,
  2,
])
def n_hid_lyr(request) -> int:
  return request.param


@pytest.fixture(params=[
  1,
  2,
])
def n_pre_hid_lyr(request) -> int:
  return request.param


@pytest.fixture(params=[
  1,
  2,
])
def n_post_hid_lyr(request) -> int:
  return request.param


@pytest.fixture(params=[
  0.0,
  0.5,
  1.0,
])
def p_emb(request) -> float:
  return request.param


@pytest.fixture(params=[
  0.0,
  0.5,
  1.0,
])
def p_hid(request) -> float:
  return request.param


@pytest.fixture(params=[
  0,
  1000,
])
def ckpt(request) -> int:
  """Test experiment checkpoint."""
  return request.param


@pytest.fixture
def tknzr() -> BaseTknzr:
  """Example tokenizer instance."""
  return CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=1,
    tk2id={
      '[bos]': 0,
      '[eos]': 1,
      '[pad]': 2,
      '[unk]': 3,
      'a': 4,
      'b': 5,
      'c': 6,
    }
  )


@pytest.fixture
def clean_model(request, ckpt: int, exp_name: str) -> str:
  """Clean model parameters output file and directories."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_file_path = os.path.join(abs_dir_path, BaseModel.file_name.format(ckpt))

  def remove():
    if os.path.exists(abs_file_path):
      os.remove(abs_file_path)

    if os.path.exists(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(remove)
