"""Test construction utilities for all language models.

Test target:
- :py:meth:`lmp.util.model.create`.
"""

import math

import lmp.util.model
from lmp.model import LSTM1997, LSTM2000, LSTM2002, ElmanNet, TransEnc
from lmp.tknzr import BaseTknzr


def test_create_elman_net(
  d_emb: int,
  d_hid: int,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
  n_lyr: int,
  p_emb: float,
  p_hid: float,
  tknzr: BaseTknzr,
) -> None:
  """Test construction for :py:class:`lmp.model.ElmanNet`."""
  model = lmp.util.model.create(
    d_emb=d_emb,
    d_hid=d_hid,
    init_lower=init_lower,
    init_upper=init_upper,
    label_smoothing=label_smoothing,
    n_lyr=n_lyr,
    p_emb=p_emb,
    p_hid=p_hid,
    model_name=ElmanNet.model_name,
    tknzr=tknzr,
  )
  assert isinstance(model, ElmanNet)
  assert model.d_emb == d_emb
  assert model.d_hid == d_hid
  assert math.isclose(model.init_lower, init_lower)
  assert math.isclose(model.init_upper, init_upper)
  assert math.isclose(model.label_smoothing, label_smoothing)
  assert model.n_lyr == n_lyr
  assert math.isclose(model.p_emb, p_emb)
  assert math.isclose(model.p_hid, p_hid)
  assert model.emb.num_embeddings == tknzr.vocab_size


def test_create_lstm_1997(
  d_blk: int,
  d_emb: int,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
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
    init_lower=init_lower,
    init_upper=init_upper,
    label_smoothing=label_smoothing,
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
  assert math.isclose(model.init_lower, init_lower)
  assert math.isclose(model.init_upper, init_upper)
  assert math.isclose(model.label_smoothing, label_smoothing)
  assert model.n_blk == n_blk
  assert model.n_lyr == n_lyr
  assert math.isclose(model.p_emb, p_emb)
  assert math.isclose(model.p_hid, p_hid)
  assert model.emb.num_embeddings == tknzr.vocab_size


def test_create_lstm_2000(
  d_blk: int,
  d_emb: int,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
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
    init_lower=init_lower,
    init_upper=init_upper,
    label_smoothing=label_smoothing,
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
  assert math.isclose(model.init_lower, init_lower)
  assert math.isclose(model.init_upper, init_upper)
  assert math.isclose(model.label_smoothing, label_smoothing)
  assert model.n_blk == n_blk
  assert model.n_lyr == n_lyr
  assert math.isclose(model.p_emb, p_emb)
  assert math.isclose(model.p_hid, p_hid)
  assert model.emb.num_embeddings == tknzr.vocab_size


def test_create_lstm_2002(
  d_blk: int,
  d_emb: int,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
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
    init_lower=init_lower,
    init_upper=init_upper,
    label_smoothing=label_smoothing,
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
  assert math.isclose(model.init_lower, init_lower)
  assert math.isclose(model.init_upper, init_upper)
  assert math.isclose(model.label_smoothing, label_smoothing)
  assert model.n_blk == n_blk
  assert model.n_lyr == n_lyr
  assert math.isclose(model.p_emb, p_emb)
  assert math.isclose(model.p_hid, p_hid)
  assert model.emb.num_embeddings == tknzr.vocab_size


def test_create_trans_enc(
  d_ff: int,
  d_k: int,
  d_model: int,
  d_v: int,
  init_lower: float,
  init_upper: float,
  label_smoothing: float,
  max_seq_len: int,
  n_head: int,
  n_lyr: int,
  p_hid: float,
  tknzr: BaseTknzr,
) -> None:
  """Test construction for :py:class:`lmp.model.TransEnc`."""
  model = lmp.util.model.create(
    d_ff=d_ff,
    d_k=d_k,
    d_model=d_model,
    d_v=d_v,
    init_lower=init_lower,
    init_upper=init_upper,
    label_smoothing=label_smoothing,
    max_seq_len=max_seq_len,
    n_head=n_head,
    n_lyr=n_lyr,
    p=p_hid,
    model_name=TransEnc.model_name,
    tknzr=tknzr,
  )
  assert isinstance(model, TransEnc)
  assert model.d_ff == d_ff
  assert model.d_k == d_k
  assert model.d_model == d_model
  assert model.d_v == d_v
  assert math.isclose(model.init_lower, init_lower)
  assert math.isclose(model.init_upper, init_upper)
  assert math.isclose(model.label_smoothing, label_smoothing)
  assert model.max_seq_len == max_seq_len
  assert model.n_head == n_head
  assert model.n_lyr == n_lyr
  assert math.isclose(model.p, p_hid)
  assert model.emb.num_embeddings == tknzr.vocab_size
