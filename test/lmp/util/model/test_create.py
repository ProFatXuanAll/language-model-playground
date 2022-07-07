"""Test construction utilities for all language models.

Test target:
- :py:meth:`lmp.util.model.create`.
"""

import math

import torch

import lmp.util.model
from lmp.model import LSTM1997, LSTM2000, LSTM2002, ElmanNet
from lmp.tknzr import BaseTknzr


def test_create_elman_net(d_emb: int, d_hid: int, p_emb: float, p_hid: float, tknzr: BaseTknzr) -> None:
  """Test construction for :py:class:`lmp.model.ElmanNet`."""
  model = lmp.util.model.create(
    d_emb=d_emb,
    d_hid=d_hid,
    p_emb=p_emb,
    p_hid=p_hid,
    model_name=ElmanNet.model_name,
    tknzr=tknzr,
  )
  assert isinstance(model, ElmanNet)
  assert model.emb.embedding_dim == d_emb
  assert model.emb.num_embeddings == tknzr.vocab_size
  assert math.isclose(model.fc_e2h[0].p, p_emb)
  assert model.fc_e2h[1].weight.size() == torch.Size([d_hid, d_emb])
  assert math.isclose(model.fc_h2e[0].p, p_hid)


def test_create_lstm_1997(d_blk: int, d_emb: int, n_blk: int, p_emb: float, p_hid: float, tknzr: BaseTknzr) -> None:
  """Test construction for :py:class:`lmp.model.LSTM1997`."""
  model = lmp.util.model.create(
    d_blk=d_blk,
    d_emb=d_emb,
    model_name=LSTM1997.model_name,
    n_blk=n_blk,
    p_emb=p_emb,
    p_hid=p_hid,
    tknzr=tknzr,
  )
  assert isinstance(model, LSTM1997)
  assert model.emb.embedding_dim == d_emb
  assert model.emb.num_embeddings == tknzr.vocab_size
  assert math.isclose(model.fc_e2ig[0].p, p_emb)
  assert model.c_0.size() == torch.Size([1, n_blk, d_blk])
  assert math.isclose(model.fc_h2e[0].p, p_hid)


def test_create_lstm_2000(d_blk: int, d_emb: int, n_blk: int, p_emb: float, p_hid: float, tknzr: BaseTknzr) -> None:
  """Test construction for :py:class:`lmp.model.LSTM2000`."""
  model = lmp.util.model.create(
    d_blk=d_blk,
    d_emb=d_emb,
    model_name=LSTM2000.model_name,
    n_blk=n_blk,
    p_emb=p_emb,
    p_hid=p_hid,
    tknzr=tknzr,
  )
  assert isinstance(model, LSTM2000)
  assert model.emb.embedding_dim == d_emb
  assert model.emb.num_embeddings == tknzr.vocab_size
  assert math.isclose(model.fc_e2ig[0].p, p_emb)
  assert model.c_0.size() == torch.Size([1, n_blk, d_blk])
  assert math.isclose(model.fc_h2e[0].p, p_hid)


def test_create_lstm_2002(d_blk: int, d_emb: int, n_blk: int, p_emb: float, p_hid: float, tknzr: BaseTknzr) -> None:
  """Test construction for :py:class:`lmp.model.LSTM2002`."""
  model = lmp.util.model.create(
    d_blk=d_blk,
    d_emb=d_emb,
    model_name=LSTM2002.model_name,
    n_blk=n_blk,
    p_emb=p_emb,
    p_hid=p_hid,
    tknzr=tknzr,
  )
  assert isinstance(model, LSTM2002)
  assert model.emb.embedding_dim == d_emb
  assert model.emb.num_embeddings == tknzr.vocab_size
  assert math.isclose(model.fc_e2ig[0].p, p_emb)
  assert model.c_0.size() == torch.Size([1, n_blk, d_blk])
  assert math.isclose(model.fc_h2e[0].p, p_hid)
