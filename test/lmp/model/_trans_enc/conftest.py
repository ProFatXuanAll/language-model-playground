"""Setup fixtures for testing :py:class:`lmp.model._trans_enc`."""

import pytest
import torch

from lmp.model._trans_enc import MultiHeadAttnLayer, PosEncLayer, TransEnc, TransEncLayer
from lmp.tknzr._base import BaseTknzr


@pytest.fixture
def multi_head_attn_layer(
  d_k: int,
  d_model: int,
  d_v: int,
  init_lower: float,
  init_upper: float,
  n_head: int,
) -> MultiHeadAttnLayer:
  """:py:class:`lmp.model._trans_enc.MultiHeadAttnLayer` instance."""
  return MultiHeadAttnLayer(
    d_k=d_k,
    d_model=d_model,
    d_v=d_v,
    init_lower=init_lower,
    init_upper=init_upper,
    n_head=n_head,
  )


@pytest.fixture
def pos_enc_layer(
  d_emb: int,
  max_seq_len: int,
) -> PosEncLayer:
  """:py:class:`lmp.model._trans_enc.PosEncLayer` instance."""
  return PosEncLayer(d_emb=d_emb, max_seq_len=max_seq_len)


@pytest.fixture
def trans_enc(
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
) -> TransEnc:
  """:py:class:`lmp.model._trans_enc.TransEnc` instance."""
  return TransEnc(
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
    tknzr=tknzr,
  )


@pytest.fixture
def trans_enc_layer(
  d_ff: int,
  d_k: int,
  d_model: int,
  d_v: int,
  init_lower: float,
  init_upper: float,
  n_head: int,
  p_hid: float,
) -> TransEncLayer:
  """:py:class:`lmp.model._trans_enc.TransEncLayer` instance."""
  return TransEncLayer(
    d_ff=d_ff,
    d_k=d_k,
    d_model=d_model,
    d_v=d_v,
    init_lower=init_lower,
    init_upper=init_upper,
    n_head=n_head,
    p=p_hid,
  )


@pytest.fixture
def batch_tkids(trans_enc: TransEnc) -> torch.Tensor:
  """Batch of token ids."""
  # Shape: (2, 4).
  return torch.randint(low=0, high=trans_enc.emb.num_embeddings, size=(2, 4))


@pytest.fixture
def batch_cur_tkids(batch_tkids: torch.Tensor) -> torch.Tensor:
  """Batch of input token ids."""
  # Shape: (2, 3).
  return batch_tkids[..., :-1]


@pytest.fixture
def batch_next_tkids(batch_tkids: torch.Tensor) -> torch.Tensor:
  """Batch of target token ids."""
  # Shape: (2, 3).
  return batch_tkids[..., 1:]


@pytest.fixture
def x(trans_enc_layer: TransEncLayer) -> torch.Tensor:
  """Batch of input features."""
  # Shape: (2, 3, d_model)
  return torch.rand((2, 3, trans_enc_layer.d_model))


@pytest.fixture
def q(x: torch.Tensor) -> torch.Tensor:
  """Batch of query features."""
  # Shape: (2, 3, d_model)
  return x


@pytest.fixture
def k(q: torch.Tensor) -> torch.Tensor:
  """Batch of key features."""
  # Shape: (2, 4, d_model)
  return torch.rand((q.size(0), q.size(1) + 1, q.size(2)))


@pytest.fixture
def v(k: torch.Tensor) -> torch.Tensor:
  """Batch of value features."""
  # Shape: (2, 4, d_model)
  return torch.rand(k.size())


@pytest.fixture
def qk_mask(k: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
  """Batch of attention mask."""
  # Shape: (2, 3, 4)
  return torch.rand((q.size(0), q.size(1), k.size(1))) > 0.5


@pytest.fixture
def x_mask(x: torch.Tensor) -> torch.Tensor:
  """Batch of attention mask."""
  # Shape: (2, 3, 3)
  return torch.rand((x.size(0), x.size(1), x.size(1))) > 0.5


@pytest.fixture
def seq_len(batch_tkids: torch.Tensor) -> int:
  """Sequence length."""
  # Shape: (2, 4)
  return batch_tkids.size(1)
