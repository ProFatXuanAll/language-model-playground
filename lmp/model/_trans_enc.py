r"""Transformer language model."""

import argparse
import math
from typing import Any, ClassVar, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp.util.metric
import lmp.util.validate
from lmp.model._base import BaseModel
from lmp.tknzr._base import PAD_TKID, BaseTknzr


class MultiHeadAttnLayer(nn.Module):
  r"""Multi-head attention :footcite:`vaswani2017attention` layer.

  - Let :math:`B` be input mini-batch size.
  - Let :math:`S_q` be the length of each query sequence.
  - Let :math:`S_k` be the length of each key sequence.
  - Let :math:`\dMdl` be the number of features per time step in each sequence.
  - Let :math:`q` be a batch of query sequence with shape :math:`(B, S_q, \dMdl)`.
  - Let :math:`k` be a batch of key sequence with shape :math:`(B, S_k, \dMdl)`.
  - Let :math:`v` be a batch of value sequence with shape :math:`(B, S_k, \dMdl)`.
  - Let :math:`\msk` be a batch of attention mask with shape :math:`(B, S_q, S_k)`.
  - Let :math:`\nHd` be the number of attention heads.
  - Let :math:`d_k` be the number of key features in each attention head.
  - Let :math:`d_v` be the number of value features in each attention head.

  Multi-head attention layer is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\MultiHeadAttnLayer}(k, \msk, q, v)                          \\
      & \indent{1} S_q \algoEq q.\sz{1}                                        \\
      & \indent{1} S_k \algoEq k.\sz{1}                                        \\
      & \indent{1} \algoFor{h \in \set{1, \dots, \nHd}}                        \\
      & \indent{2} \algoCmt{Get query vector for each head.}                   \\
      & \indent{2} \algoFor{t \in \set{1, \dots, S_q}}                         \\
      & \indent{3} q_t^h \algoEq W_{Q, h} \cdot q_t                            \\
      & \indent{2} \algoEndFor                                                 \\
      & \indent{2} Q^h \algoEq \cat{q_1^h, \dots, q_{S_q}^h}                   \\
      & \indent{2} \algoCmt{Get key-value vectors for each head.}              \\
      & \indent{2} \algoFor{t \in \set{1, \dots, S_k}}                         \\
      & \indent{3} k_t^h \algoEq W_{K, h} \cdot k_t                            \\
      & \indent{3} v_t^h \algoEq W_{V, h} \cdot v_t                            \\
      & \indent{2} \algoEndFor                                                 \\
      & \indent{2} K^h \algoEq \cat{k_1^h, \dots, k_{S_k}^h}                   \\
      & \indent{2} V^h \algoEq \cat{v_1^h, \dots, v_{S_k}^h}                   \\
      & \indent{2} \algoCmt{Apply attention mask on similarity scores.}        \\
      & \indent{2} \Sim^h \algoEq \dfrac{Q^h \cdot \pa{K^h}^\top}{\sqrt{d_k}}  \\
      & \indent{2} \algoFor{i \in \set{1, \dots, S_q}}                         \\
      & \indent{3} \algoFor{j \in \set{1, \dots, S_k}}                         \\
      & \indent{4} \algoIf{\msk_{i,j} \algoIs \algoTrue}                       \\
      & \indent{5} \Sim_{i,j}^h \algoEq -10^9                                  \\
      & \indent{4} \algoEndIf                                                  \\
      & \indent{3} \algoEndFor                                                 \\
      & \indent{3} \attn_i^h \algoEq \sof{\Sim_{i,1}^h, \dots, \Sim_{i,S_k}^h} \\
      & \indent{2} \algoEndFor                                                 \\
      & \indent{2} \algoCmt{Get attention scores.}                             \\
      & \indent{2} \attn^h \algoEq \cat{\attn_1^h, \dots, \attn_{S_q}^h}       \\
      & \indent{2} F^h \algoEq \attn^h \cdot V^h                               \\
      & \indent{1} \algoEndFor                                                 \\
      & \indent{1} F \algoEq \fla{F^1, \dots, F^{\nHd}}                        \\
      & \indent{1} O \algoEq W_O \cdot F                                       \\
      & \indent{1} \algoReturn O                                               \\
      & \algoEndProc
    \end{align*}

  +-----------------------------------------------------+----------------------------------------------------------+
  | Trainable Parameters                                | Nodes                                                    |
  +------------------+----------------------------------+----------------------+-----------------------------------+
  | Parameter        | Shape                            | Symbol               | Shape                             |
  +==================+==================================+======================+===================================+
  | :math:`W_{K,h}`  | :math:`(d_k, \dMdl)`             | :math:`F`            | :math:`(B, S_q, \nHd \times d_v)` |
  +------------------+----------------------------------+----------------------+-----------------------------------+
  | :math:`W_O`      | :math:`(\dMdl, \nHd \times d_v)` | :math:`F^h`          | :math:`(B, S_q, d_v)`             |
  +------------------+----------------------------------+----------------------+-----------------------------------+
  | :math:`W_{Q,h}`  | :math:`(d_k, \dMdl)`             | :math:`K^h`          | :math:`(B, S_k, d_k)`             |
  +------------------+----------------------------------+----------------------+-----------------------------------+
  | :math:`W_{V,h}`  | :math:`(d_v, \dMdl)`             | :math:`O`            | :math:`(B, S_q, \dMdl)`           |
  +------------------+----------------------------------+----------------------+-----------------------------------+
  |                                                     | :math:`Q^h`          | :math:`(B, S_q, d_k)`             |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`V^h`          | :math:`(B, S_k, d_v)`             |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`\attn^h`      | :math:`(B, S_q, S_k)`             |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`\attn_i^h`    | :math:`(B, S_k)`                  |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`k`            | :math:`(B, S_k, \dMdl)`           |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`k_t`          | :math:`(B, \dMdl)`                |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`k_t^h`        | :math:`(B, d_k)`                  |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`\msk`         | :math:`(B, S_q, S_k)`             |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`\msk_{i,j}`   | :math:`(B)`                       |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`q`            | :math:`(B, S_q, \dMdl)`           |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`q_t`          | :math:`(B, \dMdl)`                |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`q_t^h`        | :math:`(B, d_k)`                  |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`\Sim^h`       | :math:`(B, S_q, S_k)`             |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`\Sim_{i,j}^h` | :math:`(B)`                       |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`v`            | :math:`(B, S_k, \dMdl)`           |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`v_t`          | :math:`(B, \dMdl)`                |
  |                                                     +----------------------+-----------------------------------+
  |                                                     | :math:`v_t^h`        | :math:`(B, d_v)`                  |
  +-----------------------------------------------------+----------------------+-----------------------------------+

  Model parameters in Multi-head attention layer are initialized with uniform distribution
  :math:`\mathcal{U}(\init_l, \init_u)`.
  The lower bound :math:`\init_l` and upper bound :math:`\init_u` are given as hyperparameters.

  Parameters
  ----------
  d_k: int, default: 1
    Number of key features :math:`d_k` in each head.
  d_model: int, default: 1
    Number of input / output features :math:`\dMdl`.
  d_v: int, default: 1
    Number of value features :math:`d_v` in each head.
  init_lower: float, default: -0.1
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float, default: 0.1
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  n_head: int, default: 1
    Number of attention heads :math:`\nHd`.

  Attributes
  ----------
  d_k: int
    Number of key features :math:`d_k` in each head.
  d_model: int
    Number of input / output features :math:`\dMdl`.
  d_v: int
    Number of value features :math:`d_v` in each head.
  fc_ff_f2o: torch.nn.Linear
    Fully connected feed-forward layer :math:`W_O` which transform features to output.
    No biases are used.
    Input shape: :math:`(B, S_q, \nHd \times d_v)`.
    Output shape: :math:`(B, S_q, \dMdl)`.
  fc_ff_k2hk: torch.nn.Linear
    Fully connected feed-forward layer :math:`\pa{W_{K,1}, \dots, W_{K,\nHd}}` which transform key vectors to heads.
    No biases are used.
    Input shape: :math:`(B, S_k, \dMdl)`.
    Output shape: :math:`(B, S_k, \nHd \times d_k)`.
  fc_ff_q2hq: torch.nn.Linear
    Fully connected feed-forward layer :math:`\pa{W_{Q,1}, \dots, W_{Q,\nHd}}` which transform query vectors to heads.
    No biases are used.
    Input shape: :math:`(B, S_q, \dMdl)`.
    Output shape: :math:`(B, S_q, \nHd \times d_k)`.
  fc_ff_v2hv: torch.nn.Linear
    Fully connected feed-forward layer :math:`\pa{W_{V,1}, \dots, W_{V,\nHd}}` which transform value vectors to heads.
    No biases are used.
    Input shape: :math:`(B, S_k, \dMdl)`.
    Output shape: :math:`(B, S_k, \nHd \times d_v)`.
  init_lower: float
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  n_head: int
    Number of attention heads :math:`\nHd`.
  scaler: float
    Dot product scaler :math:`\dfrac{1}{\sqrt{d_k}}`.
  """

  def __init__(
    self,
    *,
    d_k: int = 1,
    d_model: int = 1,
    d_v: int = 1,
    init_lower: float = -0.1,
    init_upper: float = 0.1,
    n_head: int = 1,
    **kwargs: Any,
  ):
    super().__init__()

    # `d_k` validation.
    lmp.util.validate.raise_if_not_instance(val=d_k, val_name='d_k', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_k], val_names=['1', 'd_k'])
    self.d_k = d_k
    self.scaler = 1 / math.sqrt(d_k)

    # `d_model` validation.
    lmp.util.validate.raise_if_not_instance(val=d_model, val_name='d_model', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_model], val_names=['1', 'd_model'])
    self.d_model = d_model

    # `d_v` validation.
    lmp.util.validate.raise_if_not_instance(val=d_v, val_name='d_v', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_v], val_names=['1', 'd_v'])
    self.d_v = d_v

    # `init_lower` and `init_upper` validation.
    lmp.util.validate.raise_if_not_instance(val=init_lower, val_name='init_lower', val_type=float)
    lmp.util.validate.raise_if_not_instance(val=init_upper, val_name='init_upper', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[init_lower, init_upper], val_names=['init_lower', 'init_upper'])
    self.init_upper = init_upper
    self.init_lower = init_lower

    # `n_head` validation.
    lmp.util.validate.raise_if_not_instance(val=n_head, val_name='n_head', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_head], val_names=['1', 'n_head'])
    self.n_head = n_head

    # Fully connected Fully connected feed-forward layers which transform query, key and value vectors to heads.
    # No biases are used.
    self.fc_ff_q2hq = nn.Linear(in_features=d_model, out_features=n_head * d_k, bias=False)
    self.fc_ff_k2hk = nn.Linear(in_features=d_model, out_features=n_head * d_k, bias=False)
    self.fc_ff_v2hv = nn.Linear(in_features=d_model, out_features=n_head * d_v, bias=False)

    # Fully connected feed-forward layer which transform features to output.
    # No biases are used.
    self.fc_ff_f2o = nn.Linear(in_features=n_head * d_v, out_features=d_model, bias=False)

  def forward(
    self,
    k: torch.Tensor,
    mask: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
  ) -> torch.Tensor:
    r"""Perform multi-head attention on query, key, value.

    Below we describe the forward pass algorithm of multi-head attention layer.

    #. Let ``q`` be a batch of sequences of query vectors :math:`q`.
    #. Let ``k`` be a batch of sequences of key vectors :math:`k`.
    #. Let ``v`` be a batch of sequences of value vectors :math:`v`.
    #. Let ``mask`` be a batch of attention mask :math:`\msk`.
    #. Let ``q.size(1)`` be sequence length :math:`S_q`.
    #. Let ``k.size(1)`` be sequence length :math:`S_k`.
    #. Use ``self.fc_ff_q2hq`` to transform query vectors into multi-head query vectors :math:`Q^1, \dots, Q^{\nHd}`.
    #. Use ``self.fc_ff_k2hk`` to transform query vectors into multi-head query vectors :math:`K^1, \dots, K^{\nHd}`.
    #. Use ``self.fc_ff_v2hv`` to transform query vectors into multi-head query vectors :math:`V^1, \dots, V^{\nHd}`.
    #. Use :math:`Q^1, \dots, Q^{\nHd}` and :math:`K^1, \dots, K^{\nHd}` to calculate similarity scores
       :math:`\Sim^1, \dots, \Sim^{\nHd}`.
    #. Use ``mask`` to mask similarity scores :math:`\Sim^1, \dots, \Sim^{\nHd}`.
    #. Use softmax to transform similarity scores :math:`\Sim^1, \dots, \Sim^{\nHd}` into attention scores
       :math:`\attn^1, \dots, \attn^{\nHd}`.
    #. Use attention scores :math:`\attn^1, \dots, \attn^{\nHd}` and :math:`V^1, \dots, V^{\nHd}` to calculate hidden
       features :math:`F^1, \dots, F^{\nHd}`.
    #. Use :math:`W_O` and hidden features :math:`F^1, \dots, F^{\nHd}` to calculate output :math:`O`.
    #. Return :math:`O`.

    Parameters
    ----------
    k: torch.Tensor
      Batch of sequences of key vectors with shape :math:`(B, S_k, \dMdl)` and ``dtype == torch.float``.
    mask: torch.Tensor
      Batch of attention mask with shape :math:`(B, S_q, S_k)` and ``dtype == torch.bool``.
      Set to true to mask attention at corresponding position.
    q: torch.Tensor
      Batch of sequences of query vectors with shape :math:`(B, S_q, \dMdl)` and ``dtype == torch.float``.
    v: torch.Tensor
      Batch of sequences of key vectors with shape :math:`(B, S_k, \dMdl)` and ``dtype == torch.float``.

    Returns
    -------
    torch.Tensor
      Batch output features :math:`O` with shape :math:`(B, S_q, \dMdl)` and ``dtype == torch.float``.
    """
    B = q.size(0)
    S_q = q.size(1)
    S_k = k.size(1)

    # Shape: (B, n_head, S_q, d_k).
    head_q = self.fc_ff_q2hq(q).reshape(B, S_q, self.n_head, self.d_k).transpose(1, 2)

    # Shape: (B, n_head, d_k, S_k).
    head_k_T = self.fc_ff_k2hk(k).reshape(B, S_k, self.d_k, self.n_head).transpose(1, 3)

    # Shape: (B, n_head, S_k, d_v).
    head_v = self.fc_ff_v2hv(v).reshape(B, S_k, self.n_head, self.d_v).transpose(1, 2)

    # Shape: (B, n_head, S_q, S_k).
    sim = self.scaler * (head_q @ head_k_T)

    # Shape: (B, n_head, S_q, S_k).
    sim.masked_fill_(mask.unsqueeze(1), -1e9)

    # Shape: (B, n_head, S_q, S_k).
    attn = F.softmax(sim, dim=3)

    # Shape: (B, n_head, S_q, d_v).
    weighted_feat = attn @ head_v

    # Shape: (B, S_q, d_model).
    return self.fc_ff_f2o(weighted_feat.transpose(1, 2).reshape(B, S_q, self.n_head * self.d_v))

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights are initialized with uniform distribution :math:`\mathcal{U}\pa{\init_l, \init_u}`.

    Returns
    -------
    None
    """
    nn.init.uniform_(self.fc_ff_q2hq.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_ff_k2hk.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_ff_v2hv.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_ff_f2o.weight, self.init_lower, self.init_upper)


class PosEncLayer(nn.Module):
  r"""Positional Encoding :footcite:`vaswani2017attention`.

  - Let :math:`S` be the lookup sequence length.
  - Let :math:`\dEmb` be the dimension of positional encodings.

  Positional encodings is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\PosEncLayer}\pa{S}                                              \\
      & \indent{1} \algoFor{\pos \in \set{1, \dots, S}}                            \\
      & \indent{2} \algoFor{i \in \set{1, \dots, \dEmb}}                           \\
      & \indent{3} \algoIf{i \text{ is even}}                                      \\
      & \indent{4} \PE_{(\pos,i)} \algoEq \sin\pa{\dfrac{\pos}{10000^{i / \dEmb}}} \\
      & \indent{3} \algoElse                                                       \\
      & \indent{4} \PE_{(\pos,i)} \algoEq \cos\pa{\dfrac{\pos}{10000^{i / \dEmb}}} \\
      & \indent{3} \algoEndIf                                                      \\
      & \indent{2} \algoEndFor                                                     \\
      & \indent{1} \algoEndFor                                                     \\
      & \indent{1} \algoReturn \PE                                                 \\
      & \algoEndProc
    \end{align*}

  +----------------------+------------------------------------------------+
  | Trainable Parameters | Nodes                                          |
  +-------------+--------+------------------------+-----------------------+
  | Parameter   | Shape  | Symbol                 | Shape                 |
  +=============+========+========================+=======================+
  |                      | :math:`\PE_{(\pos,i)}` | :math:`(1)`           |
  |                      +------------------------+-----------------------+
  |                      | :math:`\PE`            | :math:`(1, S, \dEmb)` |
  +----------------------+------------------------+-----------------------+

  Parameters
  ----------
  d_emb: int, default: 1
    Positional encoding dimension :math:`\dEmb`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  max_seq_len: int, default: 512
    Maximum length constraint on the input sequence.

  Attributes
  ----------
  d_emb: int
    Positional encoding dimension :math:`\dEmb`.
  max_seq_len: int
    Maximum length constraint on the input sequence.
  pe: torch.Tensor
    Positional encoding lookup table.
  """

  def __init__(
    self,
    *,
    d_emb: int = 1,
    max_seq_len: int = 512,
    **kwargs: Any,
  ):
    super().__init__()

    # `d_emb` validation.
    lmp.util.validate.raise_if_not_instance(val=d_emb, val_name='d_emb', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_emb], val_names=['1', 'd_emb'])
    self.d_emb = d_emb

    # `max_seq_len` validation.
    lmp.util.validate.raise_if_not_instance(val=max_seq_len, val_name='max_seq_len', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, max_seq_len], val_names=['1', 'max_seq_len'])
    self.max_seq_len = max_seq_len

    # Create positional encoding lookup table.
    # Shape: `(S, d_emb)`.
    pe = torch.zeros((max_seq_len, d_emb))

    # Position order from `0` to `S - 1`.
    # Shape: `(S, 1)`.
    pos = torch.arange(0, max_seq_len).unsqueeze(1)

    # Compute the positional encodings.
    # Shape: `(S, d_emb)`.
    div_term = torch.exp(torch.arange(0, d_emb, 2) * (-math.log(10000) / d_emb))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)

    # First dimension is set to `1` to so that ``self.pe`` can broadcast along batch dimension.
    pe = pe.unsqueeze(0)
    self.register_buffer(name='pe', tensor=pe, persistent=True)

  def forward(self, seq_len: int) -> torch.Tensor:
    r"""Lookup positional encodings.

    Lookup is starting from position ``0`` and end at position ``seq_len - 1`` (exclusive).

    Parameters
    ----------
    seq_len: int
      Sequence length :math:`S`.

    Returns
    -------
    torch.Tensor
      Positional encodings with shape :math:`(1, S, \dEmb)` and ``dtype == torch.float``.
    """
    # `seq_len` validation.
    lmp.util.validate.raise_if_wrong_ordered(
      vals=[seq_len, self.max_seq_len],
      val_names=['seq_len', 'self.max_seq_len'],
    )

    # Lookup positional encodings.
    # Shape: `(1, S, d_emb)`.
    return self.pe[:, :seq_len, :]

  def params_init(self) -> None:
    r"""Do nothing.

    Returns
    -------
    None
    """
    pass


class TransEnc(BaseModel):
  r"""Transformer encoder :footcite:`vaswani2017attention` language model.

  - Let :math:`x` be batch of token ids with batch size :math:`B` and per sequence length :math:`S`.
  - Let :math:`c` be previous batch of token ids (previous context window) with shape :math:`(B, S')`.
    Note that :math:`c` can be empty.
  - Let :math:`S_\max` be the maximum sequence length a model can deal with.

    - When :math:`c` is empty, the constraint :math:`S \leq S_\max` must be satisfied.
    - When :math:`c` is not empty, the constraint :math:`S + S' \leq S_\max` must be satisfied.

  - Let :math:`V` be the vocabulary size of the paired tokenizer.
    Each token id represents an unique token, i.e., :math:`x_t \in \set{1, \dots, V}`.
  - Let :math:`E` be the token embedding lookup table.

    - Let :math:`\dMdl` be the dimension of token embeddings.
    - Let :math:`e_t` be the token embedding correspond to token id :math:`x_t`.
    - Token embeddings have dropout probability :math:`p`.

  - Let :math:`\PE` be positional encoding layer.

    - Let :math:`\PE_t` be the positional encoding at the :math:`t` th position.
    - The dimension of positional encodings is :math:`\dMdl`.

  - Let :math:`\nLyr` be the number of transformer encoder layers.
  - Let :math:`h^\ell` be the output of the :math:`\ell` th transformer encoder layer.

  Transformer encoder language model is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\TransEnc}\pa{x, c}                                                         \\
      & \indent{1} \algoIf{c \text{ is not empty}}                                            \\
      & \indent{2} x \algoEq \cat{x, c}                                                       \\
      & \indent{1} \algoEndIf                                                                 \\
      & \indent{1} S \algoEq x.\sz{1}                                                         \\
      & \indent{1} \algoCmt{Create attention mask.}                                           \\
      & \indent{1} \algoFor{i \in \set{1, \dots, S}}                                          \\
      & \indent{2} \algoFor{j \in \set{1, \dots, S}}                                          \\
      & \indent{3} \algoIf{x_i \algoIs \text{padding}}                                        \\
      & \indent{4} \msk_{i,j} \algoEq \algoTrue                                               \\
      & \indent{3} \algoElseIf{i \leq j}                                                      \\
      & \indent{4} \msk_{i,j} \algoEq \algoFalse                                              \\
      & \indent{3} \algoElse                                                                  \\
      & \indent{4} \msk_{i,j} \algoEq \algoTrue                                               \\
      & \indent{3} \algoEndIf                                                                 \\
      & \indent{2} \algoEndFor                                                                \\
      & \indent{1} \algoEndFor                                                                \\
      & \indent{1} \algoCmt{Lookup token embedding and positional encoding.}                  \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                          \\
      & \indent{2} e_t \algoEq (x_t)\text{-th row of } E \text{ but treated as column vector} \\
      & \indent{2} h_t^0 \algoEq \drop{e_t + \PE_t}{p}                                        \\
      & \indent{1} \algoEndFor                                                                \\
      & \indent{1} h^0 \algoEq \cat{h_1^0, \dots, h_S^0}                                      \\
      & \indent{1} \algoCmt{Perform forward pass on stacking Transformer encoder layers}      \\
      & \indent{1} \algoFor{\ell \in \set{1, \dots, \nLyr}}                                   \\
      & \indent{2} h^\ell \algoEq \TransEncLayer\pa{
                     k \algoEq h^{\ell-1},
                     \msk \algoEq \msk,
                     q \algoEq h^{\ell-1},
                     v \algoEq h^{\ell-1}
                   }                                                                          \\
      & \indent{1} \algoEndFor                                                                \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                          \\
      & \indent{2} y_t \algoEq \sof{E \cdot h_t^{\nLyr}}                                      \\
      & \indent{1} \algoEndFor                                                                \\
      & \indent{1} y \algoEq \cat{y_1, \dots, y_S}                                            \\
      & \indent{1} c' \algoEq \cat{x_{\max\pa{1, S - (S_\max-2)}}, \dots, x_S}                \\
      & \indent{1} \algoReturn \pa{y, c'}                                                     \\
      & \algoEndProc
    \end{align*}

  +-------------------------------------------+-------------------------------------------------------+
  | Trainable Parameters                      | Nodes                                                 |
  +------------------+------------------------+--------------------------+----------------------------+
  | Parameter        | Shape                  | Symbol                   | Shape                      |
  +==================+========================+==========================+============================+
  | :math:`E`        | :math:`(V, \dMdl)`     | :math:`\PE`              | :math:`(B, S_\max, \dMdl)` |
  +------------------+------------------------+--------------------------+----------------------------+
  | :math:`\TransEncLayer`                    | :math:`\PE_t`            | :math:`(B, \dMdl)`         |
  +------------------+------------------------+--------------------------+----------------------------+
  |                                           | :math:`c`                | :math:`(B, S')`            |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`c'`               | :math:`(B, S_\max-1)`      |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`e_t`              | :math:`(B, S, \dMdl)`      |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`h^\ell`           | :math:`(B, S, \dMdl)`      |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`h_t^0`            | :math:`(B, \dMdl)`         |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`\msk`             | :math:`(B, S, S)`          |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`\msk_{i,j}`       | :math:`(B)`                |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`x`                | :math:`(B, S)`             |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`x_t`              | :math:`(B)`                |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`y`                | :math:`(B, S, V)`          |
  |                                           +--------------------------+----------------------------+
  |                                           | :math:`y_t`              | :math:`(B, V)`             |
  +-------------------------------------------+--------------------------+----------------------------+

  The goal of optimization is to minimize the negative logliklihood of next token id :math:`x_{t+1}` given :math:`y_t`.
  The prediction loss is defined to be the average negative logliklihood over :math:`x` given :math:`y`.

  .. math::

    \loss = \dfrac{-1}{S} \sum_{t = 1}^S \log \Pr(x_{t+1} \vert y_t).

  - :math:`y_t` is the next token id prediction probability distribution over tokenizer's vocabulary.
    We use inner product to calculate similarity scores over all token ids, and then use softmax to normalize
    similarity scores into probability range :math:`[0, 1]`.
  - Model parameters in Transformer encoder language model are initialized with uniform distribution
    :math:`\mathcal{U}(\init_l, \init_u)`.
    The lower bound :math:`\init_l` and upper bound :math:`\init_u` of uniform distribution are given as
    hyperparameters.

  Parameters
  ----------
  d_ff: int, default: 1
    Number of hidden units :math:`\dFf` in the 2-layer fully connected feed-forward network.
  d_k: int, default: 1
    Number of key features :math:`d_k` in each head.
  d_model: int, default: 1
    Number of input / output features :math:`\dMdl`.
  d_v: int, default: 1
    Number of value features :math:`d_v` in each head.
  init_lower: float, default: -0.1
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float, default: 0.1
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  label_smoothing: float, default: 0.0
    Smoothing applied on prediction target :math:`x_{t+1}`.
  max_seq_len: int, default: 512
    Maximum length of the input sequence.
  n_lyr: int, default: 1
    Number of Transformer encoder layers :math:`\nLyr`.
  n_head: int, default: 1
    Number of attention heads :math:`\nHd`.
  p: float, default: 0.0
    Dropout probability :math:`p`.
  tknzr: ~lmp.tknzr.BaseTknzr
    Tokenizer instance.

  Attributes
  ----------
  d_ff: int
    Number of hidden units :math:`\dFf` in the 2-layer fully connected feed-forward network.
  d_k: int
    Number of key features :math:`d_k` in each head.
  d_model: int
    Number of input / output features :math:`\dMdl`.
  d_v: int
    Number of value features :math:`d_v` in each head.
  emb: torch.nn.Embedding
    Token embedding lookup matrix.
    Use token ids to lookup token embeddings.
  init_lower: float
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  input_dp: torch.nn.Dropout
    Dropout with probability :math:`p` applied on the sum of token embeddings and position encodings.
  label_smoothing: float
    Smoothing applied on prediction target :math:`x_{t+1}`.
  loss_fn: torch.nn.CrossEntropyLoss
    Loss function to be optimized.
  model_name: ClassVar[str]
    CLI name of Transformer encoder is ``Transformer-encoder``.
  p: float
    Dropout probability :math:`p`.
  pos_enc: lmp.model.PosEncLayer
    Positional Encoding.
  stack_trans_enc: torch.nn.ModuleList
    :py:class:`~TransEncLayer` stacking layers.
    The number of stacking layers is equal to :math:`\nLyr`.
    Input shape: :math:`(B, S, \dMdl)`.
    Output shape: :math:`(B, S, \dMdl)`.
  """

  model_name: ClassVar[str] = 'Transformer-encoder'

  def __init__(
    self,
    *,
    d_ff: int = 1,
    d_k: int = 1,
    d_model: int = 1,
    d_v: int = 1,
    init_lower: float = -0.1,
    init_upper: float = 0.1,
    label_smoothing: float = 0.0,
    max_seq_len: int = 512,
    n_head: int = 1,
    n_lyr: int = 1,
    p: float = 0.0,
    tknzr: BaseTknzr,
    **kwargs: Any,
  ):
    super().__init__(**kwargs)

    # `d_ff` validation.
    lmp.util.validate.raise_if_not_instance(val=d_ff, val_name='d_ff', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_ff], val_names=['1', 'd_ff'])
    self.d_ff = d_ff

    # `d_k` validation.
    lmp.util.validate.raise_if_not_instance(val=d_k, val_name='d_k', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_k], val_names=['1', 'd_k'])
    self.d_k = d_k

    # `d_model` validation.
    lmp.util.validate.raise_if_not_instance(val=d_model, val_name='d_model', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_model], val_names=['1', 'd_model'])
    self.d_model = d_model

    # `d_v` validation.
    lmp.util.validate.raise_if_not_instance(val=d_v, val_name='d_v', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_v], val_names=['1', 'd_v'])
    self.d_v = d_v

    # `init_upper` and `init_lower` validation.
    lmp.util.validate.raise_if_not_instance(val=init_upper, val_name='init_upper', val_type=float)
    lmp.util.validate.raise_if_not_instance(val=init_lower, val_name='init_lower', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[init_lower, init_upper], val_names=['init_lower', 'init_upper'])
    self.init_upper = init_upper
    self.init_lower = init_lower

    # `label_smoothing` validation.
    lmp.util.validate.raise_if_not_instance(val=label_smoothing, val_name='label_smoothing', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(
      vals=[0.0, label_smoothing, 1.0],
      val_names=['0.0', 'label_smoothing', '1.0'],
    )
    self.label_smoothing = label_smoothing

    # `max_seq_len` validation.
    lmp.util.validate.raise_if_not_instance(val=max_seq_len, val_name='max_seq_len', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, max_seq_len], val_names=['1', 'max_seq_len'])
    self.max_seq_len = max_seq_len

    # `n_head` validation.
    lmp.util.validate.raise_if_not_instance(val=n_head, val_name='n_head', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_head], val_names=['1', 'n_head'])
    self.n_head = n_head

    # `n_lyr` validation.
    lmp.util.validate.raise_if_not_instance(val=n_lyr, val_name='n_lyr', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_lyr], val_names=['1', 'n_lyr'])
    self.n_lyr = n_lyr

    # `p` validation.
    lmp.util.validate.raise_if_not_instance(val=p, val_name='p', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, p, 1.0], val_names=['0.0', 'p', '1.0'])
    self.p = p

    # Token embedding layer.
    # Use token ids to perform token embeddings lookup.
    self.emb = nn.Embedding(num_embeddings=tknzr.vocab_size, embedding_dim=d_model, padding_idx=PAD_TKID)

    # Positional encoding layer.
    # Use token ids to perform positional encoding lookup.
    self.pos_enc = PosEncLayer(d_emb=d_model, max_seq_len=max_seq_len, **kwargs)

    # Token embedding and positional encoding dropout layer.
    self.input_dp = nn.Dropout(p=p)

    # Stacking transformer encoder layers.
    self.stack_trans_enc = nn.ModuleList([])
    for _ in range(n_lyr):
      self.stack_trans_enc.append(
        TransEncLayer(
          d_ff=d_ff,
          d_k=d_k,
          d_model=d_model,
          d_v=d_v,
          init_lower=init_lower,
          init_upper=init_upper,
          n_head=n_head,
          p=p,
          **kwargs,
        )
      )

    # Loss function used to optimize language model.
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TKID, label_smoothing=label_smoothing)

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    r"""Add transformer encoder language model hyperparameters to CLI arguments parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI argument parser.

    Returns
    -------
    None

    See Also
    --------
    :doc:`lmp.script.train_model </script/train_model>`
      Language model training script.

    Examples
    --------
    >>> import argparse
    >>> import math
    >>> from lmp.model import TransEnc
    >>> parser = argparse.ArgumentParser()
    >>> TransEnc.add_CLI_args(parser)
    >>> args = parser.parse_args([
    ...   '--d_ff', '2',
    ...   '--d_k', '4',
    ...   '--d_model', '6',
    ...   '--d_v', '8',
    ...   '--init_lower', '-0.01',
    ...   '--init_upper', '0.01',
    ...   '--label_smoothing', '0.1',
    ...   '--n_head', '10',
    ...   '--n_lyr', '2',
    ...   '--p', '0.1',
    ... ])
    >>> assert args.d_ff == 2
    >>> assert args.d_k == 4
    >>> assert args.d_model == 6
    >>> assert args.d_v == 8
    >>> assert math.isclose(args.init_lower, -0.01)
    >>> assert math.isclose(args.init_upper, 0.01)
    >>> assert math.isclose(args.label_smoothing, 0.1)
    >>> assert args.n_head == 10
    >>> assert args.n_lyr == 2
    >>> assert math.isclose(args.p, 0.1)
    """
    # `parser` validation.
    lmp.util.validate.raise_if_not_instance(val=parser, val_name='parser', val_type=argparse.ArgumentParser)

    # Add hyperparameters to CLI arguments.
    group = parser.add_argument_group('Transformer encoder hyperparameters')
    group.add_argument(
      '--d_ff',
      default=1,
      help='''
      Number of hidden units in the 2-layer fully connected feed-forward network.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--d_k',
      default=1,
      help='''
      Number of key features in each head.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--d_model',
      default=1,
      help='''
      Number of input / output features.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--d_v',
      default=1,
      help='''
      Number of value features in each head.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--init_lower',
      default=-0.1,
      help='''
      Uniform distribution lower bound used to initialize model parameters.
      Default is ``-0.1``.
      ''',
      type=float,
    )
    group.add_argument(
      '--init_upper',
      default=0.1,
      help='''
      Uniform distribution lower bound used to initialize model parameters.
      Default is ``0.1``.
      ''',
      type=float,
    )
    group.add_argument(
      '--label_smoothing',
      default=0.0,
      help='''
      Label smoothing applied on cross entropy loss.
      Default is ``0.0``.
      ''',
      type=float,
    )
    group.add_argument(
      '--n_head',
      default=1,
      help='''
      Number of attention heads.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--n_lyr',
      default=1,
      help='''
      Number of transformer encoder layers.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--p',
      default=0.0,
      help='''
      Dropout probability for all layers.
      Default is ``0.0``.
      ''',
      type=float,
    )

  def cal_loss(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
    batch_prev_states: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculate language model prediction loss.

    We use cross entropy loss as our training objective.
    This method is only used for training.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_next_tkids: torch.Tensor
      Prediction target of each sample in the batch.
      ``batch_next_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[torch.Tensor], default: None
      Batch of previous token ids :math:`c`.
      The tensor represent the batch of token ids used in the previous context.
      It has shape :math:`(B, S')` and ``dtype == torch.long``.
      If given, it will be concatenated with ``batch_cur_tkids``.
      Set to ``None`` to do nothing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
      The first tensor in the tuple is the mini-batch cross-entropy loss.
      Loss tensor has shape :math:`(1)` and ``dtype == torch.float``.
      The second tensor in the tuple is a batch of the token ids used in forward pass (we denoted it as :math:`c'` in
      our definition).
      The second tensor has shape :math:`(B, \min(S, S_\max-1))` and ``dtype == torch.long``.
    """
    # Get next token id logits and last hidden states.
    # Logits shape: (B, S, V).
    # Each tensor in `batch_cur_states` has shape: (B, d_model).
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate cross-entropy loss.
    # We reshape `logits` to (B x S, V) and `batch_next_tkids` to (B x S).
    # This is needed since this is how PyTorch design its API.
    # shape: (1).
    loss = self.loss_fn(logits.reshape(-1, self.emb.num_embeddings), batch_next_tkids.reshape(-1))

    # Return loss and last hidden states.
    return (loss, batch_cur_states)

  def create_mask(self, batch_tkids: torch.Tensor) -> torch.Tensor:
    r"""Create self-attention mask for ``batch_tkids``.

    Self-attention mask is created as follow:

    #. Create auto-regressive mask by masking everything above diagnoal.
       This is needed since input token at each time step can only see input tokens at previous time steps and itself.
    #. Create padding masks by masking every positions correspond to padding tokens.
       This is needed since paddings are meaningless.

    Parameters
    ----------
    batch_tkids: torch.Tensor
      Batch of token ids with shape ``(B, S)`` and ``dtype == torch.long``.

    Returns
    -------
    torch.Tensor:
      Auto-regressive self attention mask and padding mask.
      Returned tensor has shape ``(B, S, S)`` and ``dtype == torch.bool``.
    """
    # Get batch size.
    B = batch_tkids.size(0)
    S = batch_tkids.size(1)

    # Create auto-regressive mask.
    # Shape: `(B, S, S)`.
    reg_mask = torch.ones((B, S, S), dtype=torch.bool)
    reg_mask = torch.triu(reg_mask, diagonal=1).to(batch_tkids.device)

    # Create padding mask.
    # Shape: `(B, S, S)`.
    pad_mask = (batch_tkids == PAD_TKID)
    pad_mask = torch.stack([pad_mask] * S, dim=1)
    pad_mask = pad_mask | pad_mask.transpose(1, 2)

    # Attention mask is consist of auto-regressive mask and padding mask.
    # Shape: `(B, S, S)`.
    return reg_mask | pad_mask

  def forward(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculate next token id logits.

    Logits were calculated based on previous hidden states and current input token ids.
    Use :py:meth:`~pred` to convert logits into next token id probability distribution over tokenizer's vocabulary.
    Use :py:meth:`~cal_loss` to convert logits into next token id prediction loss.
    Below we describe the forward pass algorithm of Transformer encoder language model.

    #. Use token ids to lookup token embeddings with ``self.emb``.
    #. Use sequence length to lookup positional encodings with ``self.pos_enc``.
    #. Apply dropout to the sum of token embeddings and positional encodings.
    #. Feed the result into transformer encoder layer.
       We use teacher forcing in this step when perform training, i.e., inputs are directly given instead of generated
       by model.
    #. Feed the output of previous transformer encoder layer into next transformer encoder layer until all layers have
       been used once.
    #. Perform inner product on the output of the last transformer encoder layer and token embeddings to get similarity
       scores.
    #. Return similarity scores (logits).

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[torch.Tensor], default: None
      Batch of previous token ids :math:`c`.
      The tensor represent the batch of token ids used in the previous context.
      It has shape :math:`(B, S')` and ``dtype == torch.long``.
      If given, it will be concatenated with ``batch_cur_tkids``.
      Set to ``None`` to do nothing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
      The first tensor in the tuple is the batch of next token id logits with shape :math:`(B, S, V)` and
      ``dtype == torch.float``.
      The second tensor in the tuple is a batch of the token ids used in forward pass (we denoted it as :math:`c'` in
      our definition).
      The second tensor has shape :math:`(B, \min(S, S_\max-1))` and ``dtype == torch.long``.
    """
    # Concate token ids if ``batch_prev_state is not None``.
    if batch_prev_states is None:
      x = batch_cur_tkids
    else:
      x = torch.hstack([batch_prev_states, batch_cur_tkids])

    # Token embedding lookup.
    # Shape: `(B, S, d_model)`.
    e = self.emb(x)

    # Positional encoding lookup.
    # Shape: `(B, S, d_model)`.
    pos = self.pos_enc(seq_len=x.size(1))

    # Create attention mask.
    # Shape: `(B, S, S)`
    mask = self.create_mask(batch_tkids=x)

    # Loop through each layer.
    trans_enc_lyr_in = self.input_dp(e + pos)
    for lyr in range(self.n_lyr):
      # Get `lyr`-th transformer encoder layer.
      trans_enc_lyr = self.stack_trans_enc[lyr]

      # Feed previous transformer encoder layer output to next transformer encoder layer.
      # Shape: `(B, S, d_model)`.
      trans_enc_lyr_out = trans_enc_lyr(mask=mask, x=trans_enc_lyr_in)

      # Update Transformer encoder layer's input.
      trans_enc_lyr_in = trans_enc_lyr_out

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # Shape: (B, S, V).
    sim = trans_enc_lyr_out @ self.emb.weight.transpose(0, 1)
    sim = sim[:, -batch_cur_tkids.size(1):, :]

    # Record token ids participated in the forward pass.
    # Maximum recording length is equal to ``self.max_seq_len - 1``.
    batch_cur_states = x[:, -self.max_seq_len + 1:]
    return (sim, batch_cur_states)

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases are initialized with uniform distribution :math:`\mathcal{U}(\init_l, \init_u)`.

    Returns
    -------
    None

    See Also
    --------
    ~TransEncLayer.params_init
      Transformer encoder layer parameter initialization.
    """
    nn.init.uniform_(self.emb.weight, self.init_lower, self.init_upper)

    for lyr in range(self.n_lyr):
      self.stack_trans_enc[lyr].params_init()

  @torch.no_grad()
  def pred(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculate next token id probability distribution over tokenizer's vocabulary.

    Probabilities were calculated based on previous hidden states and current input token id.
    This method must only be used for inference.
    No tensor graphs will be constructed and no gradients will be calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[torch.Tensor], default: None
      Batch of previous token ids :math:`c`.
      The tensor represent the batch of token ids used in the previous context.
      It has shape :math:`(B, S')` and ``dtype == torch.long``.
      If given, it will be concatenated with ``batch_cur_tkids``.
      Set to ``None`` to do nothing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
      The first tensor in the tuple is the batch of next token id probability distribution over the paired tokenizer's
      vocabulary.
      Probability tensor has shape :math:`(B, S, V)` and ``dtype == torch.float``.
      The second tensor in the tuple is a batch of the token ids used in forward pass (we denoted it as :math:`c'` in
      our definition).
      The second tensor has shape :math:`(B, \min(S, S_\max-1))` and ``dtype == torch.long``.
    """
    # Get next token id logits and the last hidden states.
    # Logits shape: (B, S, V).
    # Each tensor in `batch_cur_states` has shape: (B, d_model).
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate next token id probability distribution using softmax.
    # shape: (B, S, V).
    return (F.softmax(logits, dim=-1), batch_cur_states)


class TransEncLayer(nn.Module):
  r"""Transformer encoder layer :footcite:`vaswani2017attention`.

  - Let :math:`B` be mini-batch size.
  - Let :math:`S` be the length of each sequence in a mini-batch.
  - Let :math:`\dMdl` be the number of features per time step in each sequence.
  - Let :math:`x` be a batch of sequences of features with shape :math:`(B, S, \dMdl)`.
  - Let :math:`\msk` be a batch of attention mask with shape :math:`(B, S, S)`.
  - Let :math:`\nHd` be the number of attention heads.
  - Let :math:`d_k` be the number of key features in each attention head.
  - Let :math:`d_v` be the number of value features in each attention head.
  - Let :math:`\dFf` be the number of hidden units in the 2-layer fully connected feed-forward network.
  - Let :math:`p` be the dropout probability.

  Transformer encoder layer is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\TransEncLayer}(\msk, x)                                                                      \\
      & \indent{1} y_1 \algoEq \MultiHeadAttnLayer\pa{k \algoEq x, \msk \algoEq \msk, q \algoEq x, v \algoEq x} \\
      & \indent{1} y_2 \algoEq \LayerNorm_1\pa{x + \drop{y_1}{p}}                                               \\
      & \indent{1} y_3 \algoEq W_2 \cdot \max\pa{\mathbf{0}, W_1 \cdot y_2 + b_1} + b_2                         \\
      & \indent{1} y_4 \algoEq \LayerNorm_2\pa{y_2 + \drop{y_3}{p}}                                             \\
      & \indent{1} \algoReturn y_4                                                                              \\
      & \algoEndProc
    \end{align*}

  +-------------------------------------+--------------------------------------------+
  | Trainable Parameters                | Nodes                                      |
  +-------------+-----------------------+--------------------+-----------------------+
  | Parameter   | Shape                 | Symbol             | Shape                 |
  +=============+=======================+====================+=======================+
  | :math:`W_1` | :math:`(\dFf, \dMdl)` | :math:`\mathbf{0}` | :math:`(B, S, \dFf)`  |
  +-------------+-----------------------+--------------------+-----------------------+
  | :math:`W_2` | :math:`(\dMdl, \dFf)` | :math:`\msk`       | :math:`(B, S, S)`     |
  +-------------+-----------------------+--------------------+-----------------------+
  | :math:`b_1` | :math:`(\dFf)`        | :math:`x`          | :math:`(B, S, \dMdl)` |
  +-------------+-----------------------+--------------------+-----------------------+
  | :math:`b_2` | :math:`(\dMdl)`       | :math:`y_1`        | :math:`(B, S, \dMdl)` |
  +-------------+-----------------------+--------------------+-----------------------+
  | :math:`\MultiHeadAttnLayer`         | :math:`y_2`        | :math:`(B, S, \dMdl)` |
  +-------------------------------------+--------------------+-----------------------+
  | :math:`\LayerNorm_1`                | :math:`y_3`        | :math:`(B, S, \dMdl)` |
  +-------------------------------------+--------------------+-----------------------+
  | :math:`\LayerNorm_2`                | :math:`y_4`        | :math:`(B, S, \dMdl)` |
  +-------------------------------------+--------------------+-----------------------+

  Model parameters in Transformer encoder layer are initialized with uniform distribution
  :math:`\mathcal{U}(\init_l, \init_u)`.
  The lower bound :math:`\init_l` and upper bound :math:`\init_u` are given as hyperparameters.

  Parameters
  ----------
  d_ff: int
    Number of hidden units :math:`\dFf` in the 2-layer fully connected feed-forward network.
  d_k: int, default: 1
    Number of key features :math:`d_k` in each head.
  d_model: int, default: 1
    Number of input / output features :math:`\dMdl`.
  d_v: int, default: 1
    Number of value features :math:`d_v` in each head.
  init_lower: float, default: -0.1
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float, default: 0.1
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  n_head: int, default: 1
    Number of attention heads :math:`\nHd`.
  p: float, default: 0.0
    Dropout probability :math:`p`.

  Attributes
  ----------
  d_ff: int
    Number of hidden units :math:`\dFf` in the 2-layer fully connected feed-forward network.
  d_k: int
    Number of key features :math:`d_k` in each head.
  d_model: int
    Number of input / output features :math:`\dMdl`.
  d_v: int
    Number of value features :math:`d_v` in each head.
  ffn: torch.nn.Sequential
    2-layer fully connected feed-forward network with parameters :math:`W_1, W_2, b_1, b_2`.
    Dropout with probability :math:`p` is applied to output.
    Input shape: :math:`(B, S, \dMdl)`.
    Output shape: :math:`(B, S, \dMdl)`.
  init_lower: float
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  ln_1: torch.nn.LayerNorm
    Correspond to :math:`\LayerNorm_1`.
    Input shape: :math:`(B, S, \dMdl)`.
    Output shape: :math:`(B, S, \dMdl)`.
  ln_2: torch.nn.LayerNorm
    Correspond to :math:`\LayerNorm_2`.
    Input shape: :math:`(B, S, \dMdl)`.
    Output shape: :math:`(B, S, \dMdl)`.
  mha: ~MultiHeadAttnLayer
    Multi-head self attention layer.
    Multi-head attention is calculated through :math:`\MultiHeadAttnLayer` and self-attention is achieved by giving
    identical input to query, key and vector.
    Input shape: :math:`(B, S, \dMdl)`.
    Output shape: :math:`(B, S, \dMdl)`.
  mha_dp: torch.nn.Dropout
    Perform dropout with probability :math:`p` on the output of multi-head self attention.
    Input shape: :math:`(B, S, \dMdl)`.
    Output shape: :math:`(B, S, \dMdl)`.
  n_head: int
    Number of attention heads :math:`\nHd`.
  p: float
    Dropout probability :math:`p`.

  See Also
  --------
  ~MultiHeadAttnLayer
    Multi-head attention layer.
  """

  def __init__(
    self,
    *,
    d_ff: int = 1,
    d_k: int = 1,
    d_model: int = 1,
    d_v: int = 1,
    init_lower: float = -0.1,
    init_upper: float = 0.1,
    n_head: int = 1,
    p: float = 0.0,
    **kwargs: Any,
  ):
    super().__init__()

    # `d_ff` validation.
    lmp.util.validate.raise_if_not_instance(val=d_ff, val_name='d_ff', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_ff], val_names=['1', 'd_ff'])
    self.d_ff = d_ff

    # `d_k` validation.
    lmp.util.validate.raise_if_not_instance(val=d_k, val_name='d_k', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_k], val_names=['1', 'd_k'])
    self.d_k = d_k

    # `d_model` validation.
    lmp.util.validate.raise_if_not_instance(val=d_model, val_name='d_model', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_model], val_names=['1', 'd_model'])
    self.d_model = d_model

    # `d_v` validation.
    lmp.util.validate.raise_if_not_instance(val=d_v, val_name='d_v', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_v], val_names=['1', 'd_v'])
    self.d_v = d_v

    # `init_lower` and `init_upper` validation.
    lmp.util.validate.raise_if_not_instance(val=init_lower, val_name='init_lower', val_type=float)
    lmp.util.validate.raise_if_not_instance(val=init_upper, val_name='init_upper', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[init_lower, init_upper], val_names=['init_lower', 'init_upper'])
    self.init_upper = init_upper
    self.init_lower = init_lower

    # `n_head` validation.
    lmp.util.validate.raise_if_not_instance(val=n_head, val_name='n_head', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_head], val_names=['1', 'n_head'])
    self.n_head = n_head

    # `p` validation.
    lmp.util.validate.raise_if_not_instance(val=p, val_name='p', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, p, 1.0], val_names=['0.0', 'p', '1.0'])
    self.p = p

    # Multi-head attention layer.
    self.mha = MultiHeadAttnLayer(
      d_k=d_k,
      d_model=d_model,
      d_v=d_v,
      init_lower=init_lower,
      init_upper=init_upper,
      n_head=n_head,
      **kwargs,
    )

    # Dropout is applied to the output of multi-head attention layer.
    self.mha_dp = nn.Dropout(p=p)

    # 2-layer fully connected feed-forward network.
    # Dropout is applied to the output.
    self.ffn = nn.Sequential(
      nn.Linear(in_features=d_model, out_features=d_ff),
      nn.ReLU(),
      nn.Linear(in_features=d_ff, out_features=d_model),
      nn.Dropout(p=p),
    )

    # 2 different Layer Norm layer.
    self.ln_1 = nn.LayerNorm(normalized_shape=[d_model])
    self.ln_2 = nn.LayerNorm(normalized_shape=[d_model])

  def forward(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Calculate batch of hidden states for ``x``.

    Below we describe the forward pass algorithm of transformer encoder layer.

    #. Let ``x`` be a batch of sequences of features :math:`x`.
    #. Let ``mask`` be a batch of attention mask :math:`\msk`.
    #. Use ``self.mha`` to perform multi-head self attention on ``x`` and get :math:`y_1`.
    #. Use ``self.mha_dp`` to perform dropout on :math:`y_1`.
    #. Add :math:`x` and :math:`y_1` (with dropout applied) and use ``self.ln_1`` to perform layer normalization on the
       addition result to get :math:`y_2`.
    #. Use ``self.ffn`` to perform 2-layer fully connected feed-forward network forward pass and get :math:`y_3`.
    #. Add :math:`y_2` and :math:`y_3` (with dropout applied) and use ``self.ln_2`` to perform layer normalization on
       the addition result to get :math:`y_4`.
    #. Return :math:`y_4`.

    Parameters
    ----------
    x: torch.Tensor
      Batch of sequences of features with shape :math:`(B, S, \dMdl)` and ``dtype == torch.float``.
    mask: torch.Tensor
      Batch of attention mask with shape :math:`(B, S, S)` and ``dtype == torch.bool``.
      Set to true to mask attention at corresponding position.

    Returns
    -------
    torch.Tensor
      Batch of sequences of output features :math:`y_4` with shape :math:`(B, S, \dMdl)` and ``dtype == torch.float``.
    """
    # Perform multi-head self attention.
    # Shape: (B, S, d_model).
    mha_out = self.mha(q=x, k=x, v=x, mask=mask)

    # Apply dropout and residual connection, then perform layer normalization.
    # Shape: (B, S, d_model).
    x = self.ln_1(x + self.mha_dp(mha_out))

    # Feed to 2-layer fully connected feed-forward network.
    # Shape: (B, S, d_model).
    ffn_out = self.ffn(x)

    # Apply dropout and residual connection, then perform layer normalization.
    # Shape: (B, S, d_model).
    return self.ln_2(x + ffn_out)

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases are initialized with uniform distribution :math:`\mathcal{U}\pa{\init_l, \init_u}`.

    Returns
    -------
    None

    See Also
    --------
    ~MultiHeadAttnLayer.params_init
      Multi-head attention layer parameter initialization.
    """
    # Initialize multi-head attention layer.
    self.mha.params_init()

    # Initialize 2-layer fully connected feed-forward network.
    nn.init.uniform_(self.ffn[0].weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.ffn[0].bias, self.init_lower, self.init_upper)
    nn.init.uniform_(self.ffn[2].weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.ffn[2].bias, self.init_lower, self.init_upper)
