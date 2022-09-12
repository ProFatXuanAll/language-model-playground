"""Test model construction.

Test target:
- :py:meth:`lmp.model._trans_enc.MultiHeadAttnLayer.__init__`.
- :py:meth:`lmp.model._trans_enc.MultiHeadAttnLayer.params_init`.
- :py:meth:`lmp.model._trans_enc.PosEncLayer.__init__`.
- :py:meth:`lmp.model._trans_enc.PosEncLayer.params_init`.
- :py:meth:`lmp.model._trans_enc.TransEnc.__init__`.
- :py:meth:`lmp.model._trans_enc.TransEnc.params_init`.
- :py:meth:`lmp.model._trans_enc.TransEncLayer.__init__`.
- :py:meth:`lmp.model._trans_enc.TransEncLayer.params_init`.
"""

import math

import torch
import torch.nn as nn

from lmp.model._trans_enc import MultiHeadAttnLayer, PosEncLayer, TransEnc, TransEncLayer
from lmp.tknzr._base import BaseTknzr
from lmp.vars import PAD_TKID


def test_multi_head_attn_layer_default_value() -> None:
  """Ensure default value consistency."""
  d_k = 1
  d_model = 1
  d_v = 1
  init_lower = -0.1
  init_upper = 0.1
  n_head = 1
  multi_head_attn_layer = MultiHeadAttnLayer()
  multi_head_attn_layer.params_init()

  assert hasattr(multi_head_attn_layer, 'd_k')
  assert multi_head_attn_layer.d_k == d_k

  assert hasattr(multi_head_attn_layer, 'd_model')
  assert multi_head_attn_layer.d_model == d_model

  assert hasattr(multi_head_attn_layer, 'd_v')
  assert multi_head_attn_layer.d_v == d_v

  assert hasattr(multi_head_attn_layer, 'init_lower')
  assert math.isclose(multi_head_attn_layer.init_lower, init_lower)

  assert hasattr(multi_head_attn_layer, 'init_upper')
  assert math.isclose(multi_head_attn_layer.init_upper, init_upper)

  assert hasattr(multi_head_attn_layer, 'n_head')
  assert multi_head_attn_layer.n_head == n_head

  assert hasattr(multi_head_attn_layer, 'fc_ff_q2hq')
  assert isinstance(multi_head_attn_layer.fc_ff_q2hq, nn.Linear)
  assert multi_head_attn_layer.fc_ff_q2hq.weight.size() == torch.Size([n_head * d_k, d_model])
  assert multi_head_attn_layer.fc_ff_q2hq.bias is None
  assert torch.all(
    (init_lower <= multi_head_attn_layer.fc_ff_q2hq.weight) & (multi_head_attn_layer.fc_ff_q2hq.weight <= init_upper)
  )

  assert hasattr(multi_head_attn_layer, 'fc_ff_k2hk')
  assert isinstance(multi_head_attn_layer.fc_ff_k2hk, nn.Linear)
  assert multi_head_attn_layer.fc_ff_k2hk.weight.size() == torch.Size([n_head * d_k, d_model])
  assert multi_head_attn_layer.fc_ff_k2hk.bias is None
  assert torch.all(
    (init_lower <= multi_head_attn_layer.fc_ff_k2hk.weight) & (multi_head_attn_layer.fc_ff_k2hk.weight <= init_upper)
  )

  assert hasattr(multi_head_attn_layer, 'fc_ff_v2hv')
  assert isinstance(multi_head_attn_layer.fc_ff_v2hv, nn.Linear)
  assert multi_head_attn_layer.fc_ff_v2hv.weight.size() == torch.Size([n_head * d_v, d_model])
  assert multi_head_attn_layer.fc_ff_v2hv.bias is None
  assert torch.all(
    (init_lower <= multi_head_attn_layer.fc_ff_v2hv.weight) & (multi_head_attn_layer.fc_ff_v2hv.weight <= init_upper)
  )

  assert hasattr(multi_head_attn_layer, 'fc_ff_f2o')
  assert isinstance(multi_head_attn_layer.fc_ff_f2o, nn.Linear)
  assert multi_head_attn_layer.fc_ff_f2o.weight.size() == torch.Size([d_model, n_head * d_v])
  assert multi_head_attn_layer.fc_ff_f2o.bias is None
  assert torch.all(
    (init_lower <= multi_head_attn_layer.fc_ff_f2o.weight) & (multi_head_attn_layer.fc_ff_f2o.weight <= init_upper)
  )

  assert hasattr(multi_head_attn_layer, 'scaler')
  assert isinstance(multi_head_attn_layer.scaler, float)
  assert math.isclose(multi_head_attn_layer.scaler, 1 / math.sqrt(d_k))


def test_multi_head_attn_layer_parameters(
  d_k: int,
  d_model: int,
  d_v: int,
  init_lower: float,
  init_upper: float,
  multi_head_attn_layer: MultiHeadAttnLayer,
  n_head: int,
) -> None:
  """Must correctly construct parameters."""
  multi_head_attn_layer.params_init()

  assert hasattr(multi_head_attn_layer, 'd_k')
  assert multi_head_attn_layer.d_k == d_k

  assert hasattr(multi_head_attn_layer, 'd_model')
  assert multi_head_attn_layer.d_model == d_model

  assert hasattr(multi_head_attn_layer, 'd_v')
  assert multi_head_attn_layer.d_v == d_v

  assert hasattr(multi_head_attn_layer, 'init_lower')
  assert math.isclose(multi_head_attn_layer.init_lower, init_lower)

  assert hasattr(multi_head_attn_layer, 'init_upper')
  assert math.isclose(multi_head_attn_layer.init_upper, init_upper)

  assert hasattr(multi_head_attn_layer, 'n_head')
  assert multi_head_attn_layer.n_head == n_head

  assert hasattr(multi_head_attn_layer, 'fc_ff_q2hq')
  assert isinstance(multi_head_attn_layer.fc_ff_q2hq, nn.Linear)
  assert multi_head_attn_layer.fc_ff_q2hq.weight.size() == torch.Size([n_head * d_k, d_model])
  assert multi_head_attn_layer.fc_ff_q2hq.bias is None
  assert torch.all(
    (init_lower <= multi_head_attn_layer.fc_ff_q2hq.weight) & (multi_head_attn_layer.fc_ff_q2hq.weight <= init_upper)
  )

  assert hasattr(multi_head_attn_layer, 'fc_ff_k2hk')
  assert isinstance(multi_head_attn_layer.fc_ff_k2hk, nn.Linear)
  assert multi_head_attn_layer.fc_ff_k2hk.weight.size() == torch.Size([n_head * d_k, d_model])
  assert multi_head_attn_layer.fc_ff_k2hk.bias is None
  assert torch.all(
    (init_lower <= multi_head_attn_layer.fc_ff_k2hk.weight) & (multi_head_attn_layer.fc_ff_k2hk.weight <= init_upper)
  )

  assert hasattr(multi_head_attn_layer, 'fc_ff_v2hv')
  assert isinstance(multi_head_attn_layer.fc_ff_v2hv, nn.Linear)
  assert multi_head_attn_layer.fc_ff_v2hv.weight.size() == torch.Size([n_head * d_v, d_model])
  assert multi_head_attn_layer.fc_ff_v2hv.bias is None
  assert torch.all(
    (init_lower <= multi_head_attn_layer.fc_ff_v2hv.weight) & (multi_head_attn_layer.fc_ff_v2hv.weight <= init_upper)
  )

  assert hasattr(multi_head_attn_layer, 'fc_ff_f2o')
  assert isinstance(multi_head_attn_layer.fc_ff_f2o, nn.Linear)
  assert multi_head_attn_layer.fc_ff_f2o.weight.size() == torch.Size([d_model, n_head * d_v])
  assert multi_head_attn_layer.fc_ff_f2o.bias is None
  assert torch.all(
    (init_lower <= multi_head_attn_layer.fc_ff_f2o.weight) & (multi_head_attn_layer.fc_ff_f2o.weight <= init_upper)
  )

  assert hasattr(multi_head_attn_layer, 'scaler')
  assert isinstance(multi_head_attn_layer.scaler, float)
  assert math.isclose(multi_head_attn_layer.scaler, 1 / math.sqrt(d_k))


def test_pos_enc_layer_default_value() -> None:
  """Ensure default value consistency."""
  d_emb = 1
  max_seq_len = 512
  pos_enc_layer = PosEncLayer()
  pos_enc_layer.params_init()

  assert hasattr(pos_enc_layer, 'd_emb')
  assert pos_enc_layer.d_emb == d_emb

  assert hasattr(pos_enc_layer, 'max_seq_len')
  assert pos_enc_layer.max_seq_len == max_seq_len

  assert hasattr(pos_enc_layer, 'pe')
  assert isinstance(pos_enc_layer.pe, torch.Tensor)
  assert pos_enc_layer.pe.size() == torch.Size([1, max_seq_len, d_emb])
  assert torch.all((-1 <= pos_enc_layer.pe) & (pos_enc_layer.pe <= 1))


def test_pos_enc_layer_parameters(
  pos_enc_layer: PosEncLayer,
  d_emb: int,
  max_seq_len: int,
) -> None:
  """Must correctly construct parameters."""
  pos_enc_layer.params_init()

  assert hasattr(pos_enc_layer, 'd_emb')
  assert pos_enc_layer.d_emb == d_emb

  assert hasattr(pos_enc_layer, 'max_seq_len')
  assert pos_enc_layer.max_seq_len == max_seq_len

  assert hasattr(pos_enc_layer, 'pe')
  assert isinstance(pos_enc_layer.pe, torch.Tensor)
  assert pos_enc_layer.pe.size() == torch.Size([1, max_seq_len, d_emb])
  assert torch.all((-1 <= pos_enc_layer.pe) & (pos_enc_layer.pe <= 1))


def test_trans_enc_layer_default_value() -> None:
  """Ensure default value consistency."""
  d_ff = 1
  d_k = 1
  d_model = 1
  d_v = 1
  init_lower = -0.1
  init_upper = 0.1
  n_head = 1
  p = 0.0
  trans_enc_layer = TransEncLayer()
  trans_enc_layer.params_init()

  assert hasattr(trans_enc_layer, 'd_ff')
  assert trans_enc_layer.d_ff == d_ff

  assert hasattr(trans_enc_layer, 'd_k')
  assert trans_enc_layer.d_k == d_k

  assert hasattr(trans_enc_layer, 'd_model')
  assert trans_enc_layer.d_model == d_model

  assert hasattr(trans_enc_layer, 'd_v')
  assert trans_enc_layer.d_v == d_v

  assert hasattr(trans_enc_layer, 'init_lower')
  assert math.isclose(trans_enc_layer.init_lower, init_lower)

  assert hasattr(trans_enc_layer, 'init_upper')
  assert math.isclose(trans_enc_layer.init_upper, init_upper)

  assert hasattr(trans_enc_layer, 'n_head')
  assert trans_enc_layer.n_head == n_head

  assert hasattr(trans_enc_layer, 'p')
  assert math.isclose(trans_enc_layer.p, p)

  assert hasattr(trans_enc_layer, 'mha')
  assert isinstance(trans_enc_layer.mha, MultiHeadAttnLayer)
  assert trans_enc_layer.mha.d_k == d_k
  assert trans_enc_layer.mha.d_model == d_model
  assert trans_enc_layer.mha.d_v == d_v
  assert math.isclose(trans_enc_layer.mha.init_lower, init_lower)
  assert math.isclose(trans_enc_layer.mha.init_upper, init_upper)
  assert trans_enc_layer.mha.n_head == n_head

  assert hasattr(trans_enc_layer, 'mha_dp')
  assert isinstance(trans_enc_layer.mha_dp, nn.Dropout)
  assert math.isclose(trans_enc_layer.mha_dp.p, p)

  assert hasattr(trans_enc_layer, 'ffn')
  assert isinstance(trans_enc_layer.ffn, nn.Sequential)
  assert len(trans_enc_layer.ffn) == 4

  assert isinstance(trans_enc_layer.ffn[0], nn.Linear)
  assert trans_enc_layer.ffn[0].weight.size() == torch.Size([d_ff, d_model])
  assert trans_enc_layer.ffn[0].bias.size() == torch.Size([d_ff])
  assert torch.all((init_lower <= trans_enc_layer.ffn[0].weight) & (trans_enc_layer.ffn[0].weight <= init_upper))
  assert isinstance(trans_enc_layer.ffn[1], nn.ReLU)
  assert isinstance(trans_enc_layer.ffn[2], nn.Linear)
  assert trans_enc_layer.ffn[2].weight.size() == torch.Size([d_model, d_ff])
  assert trans_enc_layer.ffn[2].bias.size() == torch.Size([d_model])
  assert torch.all((init_lower <= trans_enc_layer.ffn[2].bias) & (trans_enc_layer.ffn[2].bias <= init_upper))
  assert isinstance(trans_enc_layer.ffn[3], nn.Dropout)
  assert math.isclose(trans_enc_layer.ffn[3].p, p)

  assert hasattr(trans_enc_layer, 'ln_1')
  assert isinstance(trans_enc_layer.ln_1, nn.LayerNorm)
  assert trans_enc_layer.ln_1.normalized_shape == (d_model,)

  assert hasattr(trans_enc_layer, 'ln_2')
  assert isinstance(trans_enc_layer.ln_2, nn.LayerNorm)
  assert trans_enc_layer.ln_2.normalized_shape == (d_model,)


def test_trans_enc_layer_parameters(
  trans_enc_layer: TransEncLayer,
  d_ff: int,
  d_k: int,
  d_model: int,
  d_v: int,
  init_lower: float,
  init_upper: float,
  n_head: int,
  p_hid: float,
) -> None:
  """Must correctly construct parameters."""
  trans_enc_layer.params_init()

  assert hasattr(trans_enc_layer, 'd_ff')
  assert trans_enc_layer.d_ff == d_ff

  assert hasattr(trans_enc_layer, 'd_k')
  assert trans_enc_layer.d_k == d_k

  assert hasattr(trans_enc_layer, 'd_model')
  assert trans_enc_layer.d_model == d_model

  assert hasattr(trans_enc_layer, 'd_v')
  assert trans_enc_layer.d_v == d_v

  assert hasattr(trans_enc_layer, 'init_lower')
  assert math.isclose(trans_enc_layer.init_lower, init_lower)

  assert hasattr(trans_enc_layer, 'init_upper')
  assert math.isclose(trans_enc_layer.init_upper, init_upper)

  assert hasattr(trans_enc_layer, 'n_head')
  assert trans_enc_layer.n_head == n_head

  assert hasattr(trans_enc_layer, 'p')
  assert math.isclose(trans_enc_layer.p, p_hid)

  assert hasattr(trans_enc_layer, 'mha')
  assert isinstance(trans_enc_layer.mha, MultiHeadAttnLayer)
  assert trans_enc_layer.mha.d_k == d_k
  assert trans_enc_layer.mha.d_model == d_model
  assert trans_enc_layer.mha.d_v == d_v
  assert math.isclose(trans_enc_layer.mha.init_lower, init_lower)
  assert math.isclose(trans_enc_layer.mha.init_upper, init_upper)
  assert trans_enc_layer.mha.n_head == n_head

  assert hasattr(trans_enc_layer, 'mha_dp')
  assert isinstance(trans_enc_layer.mha_dp, nn.Dropout)
  assert math.isclose(trans_enc_layer.mha_dp.p, p_hid)

  assert hasattr(trans_enc_layer, 'ffn')
  assert isinstance(trans_enc_layer.ffn, nn.Sequential)
  assert len(trans_enc_layer.ffn) == 4

  assert isinstance(trans_enc_layer.ffn[0], nn.Linear)
  assert trans_enc_layer.ffn[0].weight.size() == torch.Size([d_ff, d_model])
  assert trans_enc_layer.ffn[0].bias.size() == torch.Size([d_ff])
  assert torch.all((init_lower <= trans_enc_layer.ffn[0].weight) & (trans_enc_layer.ffn[0].weight <= init_upper))
  assert isinstance(trans_enc_layer.ffn[1], nn.ReLU)
  assert isinstance(trans_enc_layer.ffn[2], nn.Linear)
  assert trans_enc_layer.ffn[2].weight.size() == torch.Size([d_model, d_ff])
  assert trans_enc_layer.ffn[2].bias.size() == torch.Size([d_model])
  assert torch.all((init_lower <= trans_enc_layer.ffn[2].bias) & (trans_enc_layer.ffn[2].bias <= init_upper))
  assert isinstance(trans_enc_layer.ffn[3], nn.Dropout)
  assert math.isclose(trans_enc_layer.ffn[3].p, p_hid)

  assert hasattr(trans_enc_layer, 'ln_1')
  assert isinstance(trans_enc_layer.ln_1, nn.LayerNorm)
  assert trans_enc_layer.ln_1.normalized_shape == (d_model,)

  assert hasattr(trans_enc_layer, 'ln_2')
  assert isinstance(trans_enc_layer.ln_2, nn.LayerNorm)
  assert trans_enc_layer.ln_2.normalized_shape == (d_model,)


def test_trans_enc_default_value(tknzr: BaseTknzr) -> None:
  """Ensure default value consistency."""
  d_ff = 1
  d_k = 1
  d_model = 1
  d_v = 1
  init_lower = -0.1
  init_upper = 0.1
  label_smoothing = 0.0
  max_seq_len = 512
  n_head = 1
  n_lyr = 1
  p = 0.0
  trans_enc = TransEnc(tknzr=tknzr)
  trans_enc.params_init()

  assert hasattr(trans_enc, 'd_ff')
  assert trans_enc.d_ff == d_ff

  assert hasattr(trans_enc, 'd_k')
  assert trans_enc.d_k == d_k

  assert hasattr(trans_enc, 'd_model')
  assert trans_enc.d_model == d_model

  assert hasattr(trans_enc, 'd_v')
  assert trans_enc.d_v == d_v

  assert hasattr(trans_enc, 'init_lower')
  assert math.isclose(trans_enc.init_lower, init_lower)

  assert hasattr(trans_enc, 'init_upper')
  assert math.isclose(trans_enc.init_upper, init_upper)

  assert hasattr(trans_enc, 'label_smoothing')
  assert math.isclose(trans_enc.label_smoothing, label_smoothing)

  assert hasattr(trans_enc, 'n_head')
  assert trans_enc.n_head == n_head

  assert hasattr(trans_enc, 'n_lyr')
  assert trans_enc.n_lyr == n_lyr

  assert hasattr(trans_enc, 'p')
  assert math.isclose(trans_enc.p, p)

  assert hasattr(trans_enc, 'emb')
  assert isinstance(trans_enc.emb, nn.Embedding)
  assert trans_enc.emb.embedding_dim == d_model
  assert trans_enc.emb.num_embeddings == tknzr.vocab_size
  assert trans_enc.emb.padding_idx == PAD_TKID
  assert torch.all((init_lower <= trans_enc.emb.weight) & (trans_enc.emb.weight <= init_upper))

  assert hasattr(trans_enc, 'pos_enc')
  assert isinstance(trans_enc.pos_enc, PosEncLayer)
  assert trans_enc.pos_enc.d_emb == d_model
  assert trans_enc.pos_enc.max_seq_len == max_seq_len
  assert torch.all((-1 <= trans_enc.pos_enc.pe) & (trans_enc.pos_enc.pe <= 1))

  assert hasattr(trans_enc, 'input_dp')
  assert isinstance(trans_enc.input_dp, nn.Dropout)
  assert math.isclose(trans_enc.input_dp.p, p)

  assert hasattr(trans_enc, 'stack_trans_enc')
  assert isinstance(trans_enc.stack_trans_enc, nn.ModuleList)
  assert len(trans_enc.stack_trans_enc) == n_lyr
  for lyr in range(n_lyr):
    trans_enc_lyr = trans_enc.stack_trans_enc[lyr]
    assert isinstance(trans_enc_lyr, TransEncLayer)
    assert trans_enc_lyr.d_ff == d_ff
    assert trans_enc_lyr.d_k == d_k
    assert trans_enc_lyr.d_model == d_model
    assert trans_enc_lyr.d_v == d_v
    assert math.isclose(trans_enc_lyr.init_lower, init_lower)
    assert math.isclose(trans_enc_lyr.init_upper, init_upper)
    assert trans_enc_lyr.n_head == n_head
    assert math.isclose(trans_enc_lyr.p, p)

  assert hasattr(trans_enc, 'loss_fn')
  assert isinstance(trans_enc.loss_fn, nn.CrossEntropyLoss)
  assert trans_enc.loss_fn.ignore_index == PAD_TKID
  assert math.isclose(trans_enc.loss_fn.label_smoothing, label_smoothing)


def test_trans_enc_parameters(
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
  trans_enc: TransEnc,
) -> None:
  """Must correctly construct parameters."""
  trans_enc.params_init()

  assert hasattr(trans_enc, 'd_ff')
  assert trans_enc.d_ff == d_ff

  assert hasattr(trans_enc, 'd_k')
  assert trans_enc.d_k == d_k

  assert hasattr(trans_enc, 'd_model')
  assert trans_enc.d_model == d_model

  assert hasattr(trans_enc, 'd_v')
  assert trans_enc.d_v == d_v

  assert hasattr(trans_enc, 'init_lower')
  assert math.isclose(trans_enc.init_lower, init_lower)

  assert hasattr(trans_enc, 'init_upper')
  assert math.isclose(trans_enc.init_upper, init_upper)

  assert hasattr(trans_enc, 'label_smoothing')
  assert math.isclose(trans_enc.label_smoothing, label_smoothing)

  assert hasattr(trans_enc, 'n_head')
  assert trans_enc.n_head == n_head

  assert hasattr(trans_enc, 'n_lyr')
  assert trans_enc.n_lyr == n_lyr

  assert hasattr(trans_enc, 'p')
  assert math.isclose(trans_enc.p, p_hid)

  assert hasattr(trans_enc, 'emb')
  assert isinstance(trans_enc.emb, nn.Embedding)
  assert trans_enc.emb.embedding_dim == d_model
  assert trans_enc.emb.num_embeddings == tknzr.vocab_size
  assert trans_enc.emb.padding_idx == PAD_TKID
  assert torch.all((init_lower <= trans_enc.emb.weight) & (trans_enc.emb.weight <= init_upper))

  assert hasattr(trans_enc, 'pos_enc')
  assert isinstance(trans_enc.pos_enc, PosEncLayer)
  assert trans_enc.pos_enc.d_emb == d_model
  assert trans_enc.pos_enc.max_seq_len == max_seq_len
  assert torch.all((-1 <= trans_enc.pos_enc.pe) & (trans_enc.pos_enc.pe <= 1))

  assert hasattr(trans_enc, 'input_dp')
  assert isinstance(trans_enc.input_dp, nn.Dropout)
  assert math.isclose(trans_enc.input_dp.p, p_hid)

  assert hasattr(trans_enc, 'stack_trans_enc')
  assert isinstance(trans_enc.stack_trans_enc, nn.ModuleList)
  assert len(trans_enc.stack_trans_enc) == n_lyr
  for lyr in range(n_lyr):
    trans_enc_lyr = trans_enc.stack_trans_enc[lyr]
    assert isinstance(trans_enc_lyr, TransEncLayer)
    assert trans_enc_lyr.d_ff == d_ff
    assert trans_enc_lyr.d_k == d_k
    assert trans_enc_lyr.d_model == d_model
    assert trans_enc_lyr.d_v == d_v
    assert math.isclose(trans_enc_lyr.init_lower, init_lower)
    assert math.isclose(trans_enc_lyr.init_upper, init_upper)
    assert trans_enc_lyr.n_head == n_head
    assert math.isclose(trans_enc_lyr.p, p_hid)

  assert hasattr(trans_enc, 'loss_fn')
  assert isinstance(trans_enc.loss_fn, nn.CrossEntropyLoss)
  assert trans_enc.loss_fn.ignore_index == PAD_TKID
  assert math.isclose(trans_enc.loss_fn.label_smoothing, label_smoothing)
