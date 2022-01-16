"""Setup fixtures for testing :py:mod:`lmp.util.model`."""

import os

import pytest

import lmp
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def tknzr() -> BaseTknzr:
  """Example tokenizer instance."""

  return CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=1,
    tk2id={
      CharTknzr.bos_tk: CharTknzr.bos_tkid,
      CharTknzr.eos_tk: CharTknzr.eos_tkid,
      CharTknzr.pad_tk: CharTknzr.pad_tkid,
      CharTknzr.unk_tk: CharTknzr.unk_tkid,
      'a': 4,
      'b': 5,
      'c': 6,
    },
  )


@pytest.fixture
def clean_model(
  exp_name: str,
  request,
):
  """Clean up saving models."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)

  def remove():
    for file_name in os.listdir(abs_dir_path):
      abs_file_path = os.path.join(abs_dir_path, file_name)
      os.remove(abs_file_path)

    os.removedirs(abs_dir_path)

  request.addfinalizer(remove)
