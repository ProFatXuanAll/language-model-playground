"""Setup fixtures for testing :py:mod:`lmp.infer.`."""

import pytest
import torch

from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import CharTknzr
from lmp.tknzr._base import BOS_TK, EOS_TK, PAD_TK, UNK_TK, BaseTknzr


@pytest.fixture
def tknzr() -> BaseTknzr:
  """Max non special token is ``c``."""
  tknzr = CharTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  return tknzr


@pytest.fixture
def max_non_sp_tk(tknzr: BaseTknzr) -> str:
  """Max non special token (in the sense of unicode value) in tokenizer's vocabulary."""
  return max(set(tknzr.tk2id.keys()) - {BOS_TK, EOS_TK, PAD_TK, UNK_TK})


@pytest.fixture
def gen_max_non_sp_tk_model(max_non_sp_tk: str, tknzr: BaseTknzr) -> BaseModel:
  """Language model which only generates `max_non_sp_tk`."""
  model = ElmanNet(d_emb=2, d_hid=2, p_emb=0.0, p_hid=0.0, tknzr=tknzr)

  # We initialize model to make it always predict `max_non_sp_tk`.
  torch.nn.init.zeros_(model.emb.weight)
  torch.nn.init.ones_(model.emb.weight[tknzr.tk2id[max_non_sp_tk]])
  torch.nn.init.zeros_(model.fc_h2e[1].weight)
  torch.nn.init.ones_(model.fc_h2e[1].bias)

  return model
