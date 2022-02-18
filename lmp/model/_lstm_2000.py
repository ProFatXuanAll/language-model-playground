"""LSTM (2000 version) language model."""

import argparse
import math
from typing import Any, ClassVar, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp.util.metric
import lmp.util.validate
from lmp.model._base import BaseModel
from lmp.tknzr._base import PAD_TKID, BaseTknzr


class LSTM2000(BaseModel):
  r"""LSTM (2000 version) [1]_ language model.

  Implement RNN model in the paper `Learning to Forget: Continual Prediction with LSTM`_.

  - Let :math:`x` be the input token id list as defined in :py:class:`lmp.model.BaseModel`.
  - Let ``d_emb`` be the dimension of token embeddings and let ``vocab_size`` be the vocabulary size of tokenizer.
  - Let ``n_blk`` be the number of memory cell blocks and let ``d_blk`` be the dimension of each memory cell block.
  - Let :math:`\sigma` be the sigmoid function.

  LSTM (2000 version) is defined as follow:

  .. math::

     \newcommand{\pa}[1]{\left( #1 \right)}
     \newcommand{\set}[1]{\left\lbrace #1 \right\rbrace}
     \newcommand{\t}{[t]}
     \newcommand{\tp}{[t - 1]}
     \newcommand{\tz}{[0]}
     \newcommand{\c}{\operatorname{block}}
     \newcommand{\cn}[1]{{\c[#1]}}
     \newcommand{\ck}{{\cn{k}}}
     \newcommand{\nc}{{n_{\c}}}
     \newcommand{\hbar}{\overline{h}}
     \newcommand{\cat}[1]{\operatorname{concate}\pa{#1}}
     \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
     \begin{align*}
       e\t         & = (x\t)\text{-th column of } E                                                      \\
       i\t         & = \sigma\pa{W^i \cdot e\t + U^i \cdot h\tp + b^i}                                   \\
       f\t         & = \sigma\pa{W^f \cdot e\t + U^f \cdot h\tp + b^f}               && \tag{1}\label{1} \\
       o\t         & = \sigma\pa{W^o \cdot e\t + U^o \cdot h\tp + b^o}                                   \\
       k           & \in \set{1, 2, \dots, \nc}                                                          \\
       g^\ck\t     & = \tanh\pa{W^\ck \cdot e\t + U^\ck \cdot h\tp + b^\ck}                              \\
       c^\ck\t     & = f_k\t \cdot c^\ck\tp + i_k\t \cdot g^\ck\t                    && \tag{2}\label{2} \\
       \hbar^\ck\t & = o_k\t \cdot \tanh\pa{c^\ck\t}                                                     \\
       h\t         & = \cat{\hbar^\cn{1}\t, \hbar^\cn{2}\t, \dots, \hbar^\cn{\nc}\t}                     \\
       z\t         & = \tanh\pa{W^z \cdot h\t + b^z}                                                     \\
       y\t         & = \sof{E^{\top} \cdot z\t}
     \end{align*}

  +-----------------------------------------------+-------------------------------------------+
  | Trainable Parameters                          | Nodes                                     |
  +------------------+----------------------------+---------------------+---------------------+
  | Parameter        | Shape                      | Symbol              | Shape               |
  +==================+============================+=====================+=====================+
  | :math:`E`        | ``(d_emb, vocab_size)``    | :math:`e\t`         | ``(d_emb)``         |
  +------------------+----------------------------+---------------------+---------------------+
  | :math:`h\tz`     | ``(n_blk x d_blk)``        | :math:`i\t`,        | ``(n_blk)``         |
  +------------------+----------------------------+ :math:`f\t`,        |                     |
  | :math:`W^i`,     | ``(n_blk, d_emb)``         | :math:`o\t`         |                     |
  | :math:`W^f`,     |                            |                     |                     |
  | :math:`W^o`      |                            |                     |                     |
  +------------------+----------------------------+---------------------+---------------------+
  | :math:`U^i`,     | ``(n_blk, n_blk x d_blk)`` | :math:`i_k\t`,      | ``(1)``             |
  | :math:`U^f`,     |                            | :math:`f_k\t`,      |                     |
  | :math:`U^o`      |                            | :math:`o_k\t`       |                     |
  +------------------+----------------------------+---------------------+---------------------+
  | :math:`b^i`,     | ``(n_blk)``                | :math:`g^\ck\t`,    | ``(d_blk)``         |
  | :math:`b^f`,     |                            | :math:`c^\ck\t`,    |                     |
  | :math:`b^o`      |                            | :math:`\hbar^\ck\t` |                     |
  +------------------+----------------------------+---------------------+---------------------+
  | :math:`W^\ck`    | ``(d_blk, d_emb)``         | :math:`h\t`         | ``(n_blk x d_blk)`` |
  +------------------+----------------------------+---------------------+---------------------+
  | :math:`U^\ck`    | ``(d_blk, n_blk x d_blk)`` | :math:`z\t`         | ``(d_emb)``         |
  +------------------+----------------------------+---------------------+---------------------+
  | :math:`b^\ck`    | ``(d_blk)``                | :math:`y\t`         | ``(vocab_size)``    |
  +------------------+----------------------------+---------------------+---------------------+
  | :math:`c^\ck\tz` | ``(d_blk)``                |                                           |
  +------------------+----------------------------+                                           |
  | :math:`W^z`      | ``(d_emb, n_blk x d_blk)`` |                                           |
  +------------------+----------------------------+                                           |
  | :math:`b^z`      | ``(d_emb)``                |                                           |
  +------------------+----------------------------+---------------------+---------------------+

  - The only differences between :py:class:`lmp.model.LSTM1997` and :py:class:`lmp.model.LSTM2000` are equations
    :math:`\eqref{1}\eqref{2}`.
  - :math:`f\t` are forget gates at time step :math:`t`.  All other symbols have the exact same meaning as defined in
    :py:class:`lmp.model.LSTM1997`.

  Parameters
  ----------
  d_blk: int
    Dimension of each memory cell block.
  d_emb: int
    Token embedding dimension.
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.
  n_blk: int
    Number of memory cell blocks.
  p_emb: float
    Embeddings dropout probability.
  p_hid: float
    Hidden units dropout probability.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance.

  Attributes
  ----------
  c_0: torch.nn.Parameter
    Initial internal states of memory cell blocks.
  d_blk: int
    Dimension of each memory cell block.
  d_hid: int
    We set ``self.d_hid = self.n_blk * self.d_blk`` for convenience.
  emb: torch.nn.Embedding
    Token embedding lookup table.
  fg_range: tuple[int, int]
    Index range of forget gate units.
  g_range: tuple[int, int]
    Index range of all gate units.
  h_0: torch.nn.Parameter
    Initial hidden states.
  ig_range: tuple[int, int]
    Index range of input gate units.
  mc_range: tuple[int, int]
    Index range of memory cell blocks.
  model_name: ClassVar[str]
    CLI name of LSTM (2000 version) is ``LSTM-2000``.
  n_blk: int
    Number of memory cell blocks.
  og_range: tuple[int, int]
    Index range of output gate units.
  proj_e2cg: torch.nn.Sequential
    Fully connected layer which connects input units to memory cell blocks and gate units.  Input dimension is
    ``d_emb``.  Output dimension is ``n_blk * (3 + d_blk)``.
  proj_h2cg: torch.nn.Linear
    Fully connected layer which connects hidden states to memory cell blocks and gate units.  Input dimension is
    ``n_blk * d_blk``.  Output dimension is ``n_blk * (3 + d_blk)``.
  proj_h2e: torch.nn.Sequential
    Fully connected layer which connects hidden states to embedding dimension.  Input dimension is ``n_blk * d_blk``.
    Output dimension is ``d_emb``.

  See Also
  --------
  :doc:`lmp.model.BaseModel </model/BaseModel>`
    Language model utilities.
  :doc:`lmp.model.LSTM1997 </model/LSTM1997>`
    LSTM (1997 version) language model.

  References
  ----------
  .. [1] Felix A. Gers, Jürgen Schmidhuber, Fred Cummins; `Learning to Forget: Continual Prediction with LSTM`_.
         Neural Comput 2000; 12 (10): 2451--2471. doi: https://doi.org/10.1162/089976600300015015

  .. _`Learning to Forget: Continual Prediction with LSTM`:
     https://direct.mit.edu/neco/article-abstract/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM
  """

  model_name: ClassVar[str] = 'LSTM-2000'

  def __init__(
    self,
    *,
    d_blk: int,
    d_emb: int,
    n_blk: int,
    p_emb: float,
    p_hid: float,
    tknzr: BaseTknzr,
    **kwargs: Any,
  ):
    super().__init__(**kwargs)

    # `d_blk` validation.
    lmp.util.validate.raise_if_not_instance(val=d_blk, val_name='d_blk', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_blk], val_names=['1', 'd_blk'])
    self.d_blk = d_blk

    # `d_emb` validation.
    lmp.util.validate.raise_if_not_instance(val=d_emb, val_name='d_emb', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_emb], val_names=['1', 'd_emb'])

    # `n_blk` validation.
    lmp.util.validate.raise_if_not_instance(val=n_blk, val_name='n_blk', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_blk], val_names=['1', 'n_blk'])
    self.n_blk = n_blk
    self.d_hid = n_blk * d_blk

    # `p_emb` validation.
    lmp.util.validate.raise_if_not_instance(val=p_emb, val_name='p_emb', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, p_emb, 1.0], val_names=['0.0', 'p_emb', '1.0'])

    # `p_hid` validation.
    lmp.util.validate.raise_if_not_instance(val=p_hid, val_name='p_hid', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, p_hid, 1.0], val_names=['0.0', 'p_hid', '1.0'])

    # `tknzr` validation.
    lmp.util.validate.raise_if_not_instance(val=tknzr, val_name='tknzr', val_type=BaseTknzr)

    # Token embedding layer.  Use token ids to perform token embeddings lookup.
    self.emb = nn.Embedding(num_embeddings=tknzr.vocab_size, embedding_dim=d_emb, padding_idx=PAD_TKID)

    # Fully connected layer which connects input units to memory cell blocks and gate units.
    self.proj_e2cg = nn.Sequential(
      nn.Dropout(p=p_emb),
      nn.Linear(in_features=d_emb, out_features=n_blk * (3 + d_blk)),
    )

    # Define index meaning of the output of `self.proj_e2cg`.
    self.fg_range = (0, n_blk)
    self.ig_range = (self.fg_range[1], self.fg_range[1] + n_blk)
    self.og_range = (self.ig_range[1], self.ig_range[1] + n_blk)
    self.g_range = (self.fg_range[0], self.og_range[1])
    self.mc_range = (self.g_range[1], self.g_range[1] + self.d_hid)

    # Fully connected layer which connects hidden states to memory cell blocks and gate units.
    self.proj_h2cg = nn.Linear(in_features=self.d_hid, out_features=n_blk * (3 + d_blk), bias=False)

    # Initial hidden states and initial memory cell internal states.  First dimension is set to `1` to broadcast along
    # batch dimension.
    self.h_0 = nn.Parameter(torch.zeros(1, self.d_hid))
    self.c_0 = nn.Parameter(torch.zeros(1, n_blk, d_blk))

    # Fully connected layer which project hidden states to embedding dimension.
    self.proj_h2e = nn.Sequential(
      nn.Dropout(p=p_hid),
      nn.Linear(in_features=self.d_hid, out_features=d_emb),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

    # Initialize model parameters.
    self.params_init()

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and non-gate units's biases are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\frac{-1}{\sqrt{v}}, \frac{1}{\sqrt{v}}}` where :math:`v =` ``max(d_emb, n_blk x d_blk)``.
    Input gate and output gate units' biases are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\frac{-1}{\sqrt{v}}, 0}`.  Forget gate units' biases are initialized with uniform
    distribution :math:`\mathcal{U}\pa{0, \frac{1}{\sqrt{v}}}`.

    Returns
    -------
    None
    """
    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(max(self.emb.embedding_dim, self.d_hid))

    nn.init.uniform_(self.emb.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_e2cg[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_e2cg[1].bias[self.mc_range[0]:self.mc_range[1]], -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_h2cg.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.h_0, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.c_0, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_h2e[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_h2e[1].bias, -inv_sqrt_dim, inv_sqrt_dim)

    # Forget gate units' biases are initialized to positive values.
    nn.init.uniform_(self.proj_e2cg[1].bias[self.fg_range[0]:self.fg_range[1]], 0.0, inv_sqrt_dim)

    # Input gate and output gate units' biases are initialized to negative values.
    nn.init.uniform_(self.proj_e2cg[1].bias[self.ig_range[0]:self.ig_range[1]], -inv_sqrt_dim, 0.0)
    nn.init.uniform_(self.proj_e2cg[1].bias[self.og_range[0]:self.og_range[1]], -inv_sqrt_dim, 0.0)

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """CLI arguments parser for training LSTM (2000 version) language model.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI arguments parser.

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
    >>> from lmp.model import LSTM2000
    >>> parser = argparse.ArgumentParser()
    >>> LSTM2000.add_CLI_args(parser)
    >>> args = parser.parse_args([
    ...   '--d_blk', '64',
    ...   '--d_emb', '100',
    ...   '--n_blk', '8',
    ...   '--p_emb', '0.5',
    ...   '--p_hid', '0.1',
    ... ])
    >>> assert args.d_blk == 64
    >>> assert args.d_emb == 100
    >>> assert args.n_blk == 8
    >>> assert math.isclose(args.p_emb, 0.5)
    >>> assert math.isclose(args.p_hid, 0.1)
    """
    # `parser` validation.
    lmp.util.validate.raise_if_not_instance(val=parser, val_name='parser', val_type=argparse.ArgumentParser)

    # Required arguments.
    group = parser.add_argument_group('LSTM (2000 version) constructor arguments')
    group.add_argument(
      '--d_blk',
      help='Dimension of each memory cell block.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--d_emb',
      help='Token embedding dimension.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--n_blk',
      help='Number of memory cell blocks.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--p_emb',
      help='Embeddings dropout probability.',
      required=True,
      type=float,
    )
    group.add_argument(
      '--p_hid',
      help='Hidden units dropout probability.',
      required=True,
      type=float,
    )

  def forward(self, batch_cur_tkids: torch.Tensor, batch_next_tkids: torch.Tensor) -> torch.Tensor:
    """Calculate language model training loss.

    This method must only be used to train model.  For inference use :py:meth:`lmp.model.LSTM2000.pred` instead.
    Forward pass algorithm is structured as follow:

    #. Use token ids to lookup token embeddings with ``self.emb``.
    #. Use ``self.proj_e2cg`` and ``self.proj_h2cg`` to calculate memory cell blocks' and gate units' input.  In this
       step we use teacher forcing, i.e., inputs are directly given instead generated by model.
    #. Calculate forget gate units, input gate units, output gate units and memory cell block's input activations.
    #. Update memory cell block's internal state.
    #. Calculate memory cell block's output.
    #. Use ``self.proj_h2e`` to project memory cell block's output to embedding dimension.
    #. Calculate similarity scores by calculating inner product over all token embeddings.
    #. Return cross-entropy loss.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of token ids which represent input token ids of all time steps.  ``batch_cur_tkids`` has shape
      ``(batch_size, seq_len)`` and ``dtype == torch.long``.
    batch_next_tkids: torch.Tensor
      Batch of token ids which represent prediction targets of all time steps.  ``batch_next_tkids`` has the same shape
      and ``dtype`` as ``batch_cur_tkids``.

    Returns
    -------
    torch.Tensor
      Cross entropy loss on next token id prediction. Returned tensor has shape ``(1)`` and ``dtype == torch.float``.
    """
    # Sequence length.
    seq_len = batch_cur_tkids.size(1)

    # Token embedding lookup and project from embedding layer to memory cell blocks and gate units.
    # In  shape: (batch_size, seq_len).
    # Out shape: (batch_size, seq_len, n_blk x (3 + d_blk)).
    e = self.proj_e2cg(self.emb(batch_cur_tkids))

    # Perform recurrent calculation for `seq_len` steps.  We use teacher forcing, i.e., the current input `e[:, i, :]`
    # is used instead of generated by model.
    h_all = []
    c_prev: Union[torch.Tensor, nn.Parameter] = self.c_0
    h_prev: Union[torch.Tensor, nn.Parameter] = self.h_0
    for i in range(seq_len):
      # Project `h_prev` from hidden states to memory cell blocks and gate units.  Then calculate memory cell blocks
      # and gate units input.
      # shape: (batch_size, n_blk x (3 + d_blk)).
      cg_in = e[:, i, :] + self.proj_h2cg(h_prev)

      # Get forget gate, input gate and output gate units.
      # shape: (batch_size, n_blk, 1)
      g = torch.sigmoid(cg_in[:, self.g_range[0]:self.g_range[1]])
      fg = g[:, self.fg_range[0]:self.fg_range[1]].unsqueeze(2)
      ig = g[:, self.ig_range[0]:self.ig_range[1]].unsqueeze(2)
      og = g[:, self.og_range[0]:self.og_range[1]].unsqueeze(2)

      # Calculate memory cell blocks input activation and reshape to separate memory cell blocks.
      # shape: (batch_size, n_blk, d_blk)
      c_in_act = torch.tanh(cg_in[:, self.mc_range[0]:self.mc_range[1]]).reshape(-1, self.n_blk, self.d_blk)

      # Calculate memory cell blocks' current internal states.
      # shape: (batch_size, n_blk, d_blk)
      c_cur = fg * c_prev + ig * c_in_act

      # Calculate memory cell blocks' current outputs and reshape to fit the shape of hidden state.
      # shape: (batch_size, n_blk x d_blk)
      h_cur = (og * torch.tanh(c_cur)).reshape(-1, self.d_hid)

      h_all.append(h_cur)

      # Update hidden states and memory cell blocks' internal states.
      c_prev = c_cur
      h_prev = h_cur

    # Stack list of tensors into single tensor.
    # In  shape: list of (batch_size, n_blk x d_blk) with length equals to `seq_len`.
    # Out shape: (batch_size, seq_len, n_blk x d_blk).
    h = torch.stack(h_all, dim=1)

    # Project from hidden states to embedding dimension.
    # shape: (batch_size, seq_len, d_emb)
    z = self.proj_h2e(h)

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # shape: (batch_size, seq_len, vocab_size).
    sim = z @ self.emb.weight.transpose(0, 1)

    # Calculate cross-entropy loss.
    # shape: (batch_size).
    loss = lmp.util.metric.cross_entropy_loss(
      batch_tkids=batch_next_tkids,
      batch_tkids_pd=F.softmax(sim, dim=2),
    )

    # Return batch average loss.
    # shape: (1).
    return torch.nanmean(loss)

  @torch.no_grad()
  def pred(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Calculate next token id probability distribution given previous hidden states and current input token id.

    This method must only be used for inference.  For training use :py:meth:`lmp.model.LSTM2000.forward` instead.  No
    tensor graphs will be constructed and no gradients will be calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.  ``batch_cur_tkids`` has shape ``(batch_size)`` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous calculation results.  Set to ``None`` to use ``[self.h_0, self.c_0]``.  ``batch_prev_states``
      must has two items, the first item will be used as previous memory cell block's output and the second item will
      be used as previous memory cell blocks' internal states.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the tensor of batch of next token id probability distribution with shape
      ``(batch_size, vocab_size)`` and ``dtype == torch.float``.  The second item in the tuple is a list of tensor
      which represent memory cell blocks' current output and memory cell blocks' current internal states.
    """
    # Use initial hidden state if `batch_prev_state is None`.
    if batch_prev_states is None:
      batch_prev_states = [self.h_0, self.c_0]

    h_prev = batch_prev_states[0]
    c_prev = batch_prev_states[1]

    # Calculate memory cell blocks and gate units input.
    # shape: (batch_size, n_blk x (3 + d_blk)).
    cg_in = self.proj_e2cg(self.emb(batch_cur_tkids)) + self.proj_h2cg(h_prev)

    # Get forget gate, input gate and output gate units.
    # shape: (batch_size, n_blk, 1)
    g = torch.sigmoid(cg_in[:, self.g_range[0]:self.g_range[1]])
    fg = g[:, self.fg_range[0]:self.fg_range[1]].unsqueeze(2)
    ig = g[:, self.ig_range[0]:self.ig_range[1]].unsqueeze(2)
    og = g[:, self.og_range[0]:self.og_range[1]].unsqueeze(2)

    # Calculate memory cells input activation and reshape to separate memory cells.
    # shape: (batch_size, n_blk, d_blk)
    c_in_act = torch.tanh(cg_in[:, self.mc_range[0]:self.mc_range[1]]).reshape(-1, self.n_blk, self.d_blk)

    # Calculate memory cell blocks' current internal states.
    # shape: (batch_size, n_blk, d_blk)
    c_cur = fg * c_prev + ig * c_in_act

    # Calculate memory cell blocks' current outputs and reshape to fit the shape of hidden state.
    # shape: (batch_size, n_blk x d_blk)
    h_cur = (og * torch.tanh(c_cur)).reshape(-1, self.d_hid)

    # Project from hidden states to embedding dimension.  Then calculate similarity scores by calculating inner product
    # over all token embeddings.
    # shape: (batch_size, vocab_size).
    sim = self.proj_h2e(h_cur) @ self.emb.weight.transpose(0, 1)

    # Calculate next token id probability distribution using softmax.
    # shape: (batch_size, vocab_size).
    return (F.softmax(sim, dim=1), [h_cur, c_cur])