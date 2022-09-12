"""Test forward pass and tensor graph.

Test target:
- :py:meth:`lmp.model._trans_enc.MultiHeadAttnLayer.forward`.
- :py:meth:`lmp.model._trans_enc.PosEncLayer.forward`.
- :py:meth:`lmp.model._trans_enc.TransEnc.forward`.
- :py:meth:`lmp.model._trans_enc.TransEncLayer.forward`.
"""

import torch

from lmp.model._trans_enc import MultiHeadAttnLayer, PosEncLayer, TransEnc, TransEncLayer


def test_multi_head_attn_layer_forward_path(
  k: torch.Tensor,
  multi_head_attn_layer: MultiHeadAttnLayer,
  q: torch.Tensor,
  qk_mask: torch.Tensor,
  v: torch.Tensor,
) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  multi_head_attn_layer = multi_head_attn_layer.train()
  multi_head_attn_layer.zero_grad()

  B = q.size(0)
  S_q = q.size(1)
  out = multi_head_attn_layer(k=k, mask=qk_mask, q=q, v=v)

  assert isinstance(out, torch.Tensor)
  assert out.size() == torch.Size([B, S_q, multi_head_attn_layer.d_model])

  logits = out.sum()
  logits.backward()

  assert logits.size() == torch.Size([])
  assert logits.dtype == torch.float
  assert hasattr(multi_head_attn_layer.fc_ff_q2hq.weight, 'grad')
  assert hasattr(multi_head_attn_layer.fc_ff_k2hk.weight, 'grad')
  assert hasattr(multi_head_attn_layer.fc_ff_v2hv.weight, 'grad')
  assert hasattr(multi_head_attn_layer.fc_ff_f2o.weight, 'grad')

  multi_head_attn_layer.zero_grad()


def test_pos_enc_layer_forward_path(
  pos_enc_layer: PosEncLayer,
  seq_len: int,
) -> None:
  """Positional encoding does not have trainable parameters."""
  # Make sure model has zero gradients at the begining.
  pos_enc_layer = pos_enc_layer.train()

  out = pos_enc_layer(seq_len)

  assert isinstance(out, torch.Tensor)
  assert out.size() == torch.Size([1, seq_len, pos_enc_layer.d_emb])


def test_trans_enc_layer_forward_path(trans_enc_layer: TransEncLayer, x: torch.Tensor, x_mask: torch.Tensor) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  trans_enc_layer = trans_enc_layer.train()
  trans_enc_layer.zero_grad()

  B = x.size(0)
  S = x.size(1)
  out = trans_enc_layer(x=x, mask=x_mask)

  assert isinstance(out, torch.Tensor)
  assert out.size() == torch.Size([B, S, trans_enc_layer.d_model])

  logits = out.sum()
  logits.backward()

  assert logits.size() == torch.Size([])
  assert logits.dtype == torch.float
  assert hasattr(trans_enc_layer.mha.fc_ff_f2o.weight, 'grad')
  assert hasattr(trans_enc_layer.ffn[0].weight, 'grad')
  assert hasattr(trans_enc_layer.ffn[0].bias, 'grad')
  assert hasattr(trans_enc_layer.ffn[2].weight, 'grad')
  assert hasattr(trans_enc_layer.ffn[2].bias, 'grad')


def test_trans_enc_forward_path(batch_cur_tkids: torch.Tensor, trans_enc: TransEnc) -> None:
  """Parameters used during forward pass must have gradients."""
  # Make sure model has zero gradients at the begining.
  trans_enc = trans_enc.train()
  trans_enc.zero_grad()

  B = batch_cur_tkids.size(0)
  batch_prev_states = None
  for idx in range(batch_cur_tkids.size(1)):
    logits, batch_cur_states = trans_enc(
      batch_cur_tkids=batch_cur_tkids[..., idx].reshape(-1, 1),
      batch_prev_states=batch_prev_states,
    )

    assert isinstance(logits, torch.Tensor)
    assert logits.size() == torch.Size([B, 1, trans_enc.emb.num_embeddings])
    assert logits.dtype == torch.float

    logits.sum().backward()
    assert hasattr(trans_enc.emb.weight, 'grad')

    for lyr in range(trans_enc.n_lyr):
      assert hasattr(trans_enc.stack_trans_enc[lyr].ffn[0].weight, 'grad')

    assert isinstance(batch_cur_states, torch.Tensor)
    batch_cur_states.size(0) == B
    if batch_prev_states is None:
      assert batch_cur_states.size(1) == 1
    else:
      assert batch_cur_states.size(1) == min(batch_prev_states.size(1) + 1, trans_enc.max_seq_len - 1)

    trans_enc.zero_grad()
    batch_prev_states = batch_cur_states
