"""Setup fixtures for testing :py:mod:`lmp.script.tknz_txt`."""

import argparse
import os

import pytest

import lmp.util.cfg
import lmp.util.path
from lmp.tknzr import BaseTknzr, CharTknzr, WsTknzr


@pytest.fixture
def tknzr_and_cfg_file_path(exp_name: str, request) -> None:
  """Clean up saved tokenizer and its configuration file."""
  abs_dir_path = os.path.join(lmp.util.path.EXP_PATH, exp_name)
  abs_cfg_file_path = os.path.join(abs_dir_path, lmp.util.cfg.CFG_NAME)
  abs_tknzr_file_path = os.path.join(abs_dir_path, BaseTknzr.file_name)

  def fin():
    if os.path.exists(abs_cfg_file_path):
      os.remove(abs_cfg_file_path)
    if os.path.exists(abs_tknzr_file_path):
      os.remove(abs_tknzr_file_path)
    if os.path.exists(abs_dir_path) and not os.listdir(abs_dir_path):
      os.removedirs(abs_dir_path)

  request.addfinalizer(fin)


@pytest.fixture
def char_tknzr(exp_name: str, request, tknzr_and_cfg_file_path) -> CharTknzr:
  """Character tokenizer example."""
  args = argparse.Namespace(
    exp_name=exp_name,
    is_uncased=True,
    max_vocab=-1,
    min_count=0,
    tknzr_name=CharTknzr.tknzr_name,
  )
  lmp.util.cfg.save(args=args, exp_name=args.exp_name)
  tknzr = CharTknzr(
    is_uncased=args.is_uncased,
    max_vocab=args.max_vocab,
    min_count=args.min_count,
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
  tknzr.save(exp_name=args.exp_name)
  return tknzr


@pytest.fixture
def ws_tknzr(exp_name: str, request, tknzr_and_cfg_file_path) -> WsTknzr:
  """Character tokenizer example."""
  args = argparse.Namespace(
    exp_name=exp_name,
    is_uncased=True,
    max_vocab=-1,
    min_count=0,
    tknzr_name=WsTknzr.tknzr_name,
  )
  lmp.util.cfg.save(args=args, exp_name=args.exp_name)
  tknzr = WsTknzr(
    is_uncased=args.is_uncased,
    max_vocab=args.max_vocab,
    min_count=args.min_count,
    tk2id={
      WsTknzr.bos_tk: WsTknzr.bos_tkid,
      WsTknzr.eos_tk: WsTknzr.eos_tkid,
      WsTknzr.pad_tk: WsTknzr.pad_tkid,
      WsTknzr.unk_tk: WsTknzr.unk_tkid,
      'a': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 1,
      'b': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 2,
      'c': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 3,
    },
  )
  tknzr.save(exp_name=args.exp_name)
  return tknzr
