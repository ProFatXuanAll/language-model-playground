"""Setup fixtures for testing :py:mod:`lmp.infer.`."""

import pytest
import torch

from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def max_seq_len() -> int:
  """Maximum sequence length."""
  return 16


@pytest.fixture
def tknzr() -> BaseTknzr:
  """Max non special token is ``c``."""
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
def max_non_sp_tk(tknzr: BaseTknzr) -> str:
  """Max non special token (in the sense of unicode value) in tokenizer's vocabulary."""
  sp_tks = [tknzr.bos_tk, tknzr.eos_tk, tknzr.pad_tk, tknzr.unk_tk]
  return max(set(tknzr.tk2id.keys()) - set(sp_tks))


@pytest.fixture
def gen_max_non_sp_tk_model(max_non_sp_tk: str, tknzr: BaseTknzr) -> BaseModel:
  """Language model which only generate `max_non_sp_tk`."""
  model = ElmanNet(d_emb=10, tknzr=tknzr)

  for tk in tknzr.tk2id.keys():
    tkid = tknzr.tk2id[tk]
    if tk == max_non_sp_tk:
      torch.nn.init.ones_(model.emb.weight[tkid])
    else:
      torch.nn.init.zeros_(model.emb.weight[tkid])

  return model
