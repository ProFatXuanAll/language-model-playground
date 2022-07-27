"""Test construction utilities for all language models.

Test target:
- :py:meth:`lmp.util.model.create`.
"""

import math

import lmp.util.model
from lmp.model import LSTM1997, LSTM2000, LSTM2002, ElmanNet
from lmp.tknzr import BaseTknzr


def test_create_elman_net(d_emb: int, d_hid: int, n_lyr: int, p_emb: float, p_hid: float, tknzr: BaseTknzr) -> None:
  """Test construction for :py:class:`lmp.model.ElmanNet`."""
  model = lmp.util.model.create(
    d_emb=d_emb,
    d_hid=d_hid,
    n_lyr=n_lyr,
    p_emb=p_emb,
    p_hid=p_hid,
    model_name=ElmanNet.model_name,
    tknzr=tknzr,
  )
  assert isinstance(model, ElmanNet)
  assert model.d_emb == d_emb
  assert model.d_hid == d_hid
  assert model.n_lyr == n_lyr
  assert math.isclose(model.p_emb, p_emb)
  assert math.isclose(model.p_hid, p_hid)
  assert model.emb.num_embeddings == tknzr.vocab_size


def test_create_lstm_1997(
  d_blk: int,
  d_emb: int,
  n_blk: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
) -> None:
  """Test construction for :py:class:`lmp.model.LSTM1997`."""
  model = lmp.util.model.create(
    d_blk=d_blk,
    d_emb=d_emb,
    model_name=LSTM1997.model_name,
    n_blk=n_blk,
    n_lyr=n_lyr,
    p_emb=p_emb,
    p_hid=p_hid,
    tknzr=tknzr,
  )
  assert isinstance(model, LSTM1997)
  assert model.d_blk == d_blk
  assert model.d_emb == d_emb
  assert model.n_blk == n_blk
  assert model.n_lyr == n_lyr
  assert math.isclose(model.p_emb, p_emb)
  assert math.isclose(model.p_hid, p_hid)
  assert model.emb.num_embeddings == tknzr.vocab_size


def test_create_lstm_2000(
  d_blk: int,
  d_emb: int,
  n_blk: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
) -> None:
  """Test construction for :py:class:`lmp.model.LSTM2000`."""
  model = lmp.util.model.create(
    d_blk=d_blk,
    d_emb=d_emb,
    model_name=LSTM2000.model_name,
    n_blk=n_blk,
    n_lyr=n_lyr,
    p_emb=p_emb,
    p_hid=p_hid,
    tknzr=tknzr,
  )
  assert isinstance(model, LSTM2000)
  assert model.d_blk == d_blk
  assert model.d_emb == d_emb
  assert model.n_blk == n_blk
  assert model.n_lyr == n_lyr
  assert math.isclose(model.p_emb, p_emb)
  assert math.isclose(model.p_hid, p_hid)
  assert model.emb.num_embeddings == tknzr.vocab_size


def test_create_lstm_2002(
  d_blk: int,
  d_emb: int,
  n_blk: int,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
) -> None:
  """Test construction for :py:class:`lmp.model.LSTM2002`."""
  model = lmp.util.model.create(
    d_blk=d_blk,
    d_emb=d_emb,
    model_name=LSTM2002.model_name,
    n_blk=n_blk,
    n_lyr=n_lyr,
    p_emb=p_emb,
    p_hid=p_hid,
    tknzr=tknzr,
  )
  assert isinstance(model, LSTM2002)
  assert model.d_blk == d_blk
  assert model.d_emb == d_emb
  assert model.n_blk == n_blk
  assert model.n_lyr == n_lyr
  assert math.isclose(model.p_emb, p_emb)
  assert math.isclose(model.p_hid, p_hid)
  assert model.emb.num_embeddings == tknzr.vocab_size
