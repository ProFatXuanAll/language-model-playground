"""Setup fixtures for testing :py:mod:`lmp.util.model`."""

import os

import pytest

import lmp
from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def tknzr() -> BaseTknzr:
  """Tokenizer example."""
  return CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=1,
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
def model(tknzr: BaseTknzr) -> BaseModel:
  """Save model example."""
  return ElmanNet(d_emb=10, tknzr=tknzr)


@pytest.fixture
def ckpt_dir_path(exp_name: str, request) -> str:
  """Clean up saving model checkpoints."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)

  def fin() -> None:
    for file_name in os.listdir(abs_dir_path):
      os.remove(os.path.join(abs_dir_path, file_name))
    os.removedirs(abs_dir_path)

  request.addfinalizer(fin)
  return abs_dir_path
