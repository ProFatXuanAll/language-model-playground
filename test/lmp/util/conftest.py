"""Setup fixtures for testing :py:mod:`lmp.util`."""

import pytest

from lmp.model import BaseModel, ElmanNet
from lmp.tknzr import BaseTknzr, CharTknzr


@pytest.fixture
def tknzr(is_uncased: bool, max_vocab: int, min_count: int) -> BaseTknzr:
  """Tokenizer example."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count)
  tknzr.build_vocab(batch_txt=['a', 'b', 'c'])
  return tknzr


@pytest.fixture
def model(d_emb: int, d_hid: int, n_lyr: int, p_emb: float, p_hid: float, tknzr: BaseTknzr) -> BaseModel:
  """Save model example."""
  return ElmanNet(d_emb=d_emb, d_hid=d_hid, n_lyr=n_lyr, p_emb=p_emb, p_hid=p_hid, tknzr=tknzr)
