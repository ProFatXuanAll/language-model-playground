"""LSTM (2002 version) language model."""

import argparse
import math
from typing import Any, ClassVar, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp.util.validate
from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr


class LSTM2002(BaseModel):
  r"""LSTM (2002 version) [1]_ language model.

  Implement RNN model in the paper `Learning Precise Timing with LSTM Recurrent Networks`_.

  - Let :math:`x` be the input token id list as defined in :py:class:`lmp.model.BaseModel`.
  - Let ``d_emb`` be the dimension of token embeddings and let ``vocab_size`` be the vocabulary size of tokenizer.
  - Let ``n_cell`` be the number of memory cells and let ``d_cell`` be the dimension of each memory cell.

  Then LSTM (2002 version) is defined as follow:

  .. math::

     \newcommand{\pa}[1]{\left( #1 \right)}
     \newcommand{\set}[1]{\left\lbrace #1 \right\rbrace}
     \newcommand{\t}{[t]}
     \newcommand{\tn}{[t + 1]}
     \newcommand{\tz}{[0]}
     \newcommand{\c}{\operatorname{cell}}
     \newcommand{\cn}[1]{{\c[#1]}}
     \newcommand{\ck}{{\cn{k}}}
     \newcommand{\nc}{{n_{\c}}}
     \newcommand{\hbar}{\overline{h}}
     \newcommand{\cat}[1]{\operatorname{concate}\pa{#1}}
     \newcommand{\sig}[1]{\operatorname{sigmoid}\pa{#1}}
     \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
     \begin{align*}
       e\t          & = (x\t)\text{-th column of } E                                               \\
       k            & \in \set{1, 2, \dots, \nc}                                                   \\
       i\tn         & = \cat{i_1\tn, \dots, i_\nc\tn}                                              \\
       i_k\tn       & = \sig{W^{ik} \cdot e\t + U^{ik} \cdot h\t + V^{ik} \cdot c^\ck\t + b^{ik}}
                    && \tag{1}\label{1}                                                            \\
       f\tn         & = \cat{f_1\tn, \dots, f_\nc\tn}                                              \\
       f_k\tn       & = \sig{W^{fk} \cdot e\t + U^{fk} \cdot h\t + V^{fk} \cdot c^\ck\t + b^{fk}}
                    && \tag{2}\label{2}                                                            \\
       g^\ck\tn     & = W^\ck \cdot e\t + U^\ck \cdot h\t + b^\ck
                    && \tag{3}\label{3}                                                            \\
       c^\ck\tn     & = f_k\tn \cdot c^\ck\t + i_k\tn \cdot g^\ck\tn                               \\
       o\tn         & = \cat{o_1\tn, \dots, o_\nc\tn}                                              \\
       o_k\tn       & = \sig{W^{ok} \cdot e\t + U^{ok} \cdot h\t + V^{ok} \cdot c^\ck\tn + b^{ok}}
                    && \tag{4}\label{4}                                                            \\
       \hbar^\ck\tn & = o_k\tn \cdot c^\ck\tn
                    && \tag{5}\label{5}                                                            \\
       h\tn         & = \cat{\hbar^\cn{1}\tn, \hbar^\cn{2}\tn, \dots, \hbar^\cn{\nc}\tn}           \\
       z\tn         & = \sig{W^z \cdot h\tn + b^z}                                                 \\
       y\tn         & = \sof{E^{\top} \cdot z\tn}
     \end{align*}

  +--------------------------------------------------+----------------------------------------------+
  | Trainable Parameters                             | Nodes                                        |
  +------------------+-------------------------------+----------------------+-----------------------+
  | Parameter        | Shape                         | Symbol               | Shape                 |
  +==================+===============================+======================+=======================+
  | :math:`E`        | ``(d_emb, vocab_size)``       | :math:`e\t`          | ``(d_emb)``           |
  +------------------+-------------------------------+----------------------+-----------------------+
  | :math:`h\tz`     | ``(n_cell x d_cell)``         | :math:`i\tn`,        | ``(n_cell)``          |
  |                  |                               | :math:`f\tn`         |                       |
  +------------------+-------------------------------+----------------------+-----------------------+
  | :math:`c^\ck\tz` | ``(d_cell)``                  | :math:`i_k\tn`,      | ``(1)``               |
  |                  |                               | :math:`f_k\tn`,      |                       |
  +------------------+-------------------------------+----------------------+-----------------------+
  | :math:`W^{ik}`,  | ``(1, d_emb)``                | :math:`g^\ck\tn`,    | ``(d_cell)``          |
  | :math:`W^{fk}`,  |                               | :math:`c^\ck\tn`,    |                       |
  | :math:`W^{ok}`   |                               |                      |                       |
  +------------------+-------------------------------+----------------------+-----------------------+
  | :math:`U^{ik}`,  | ``(1, n_cell x d_cell)``      | :math:`o\tn`         | ``(n_cell)``          |
  | :math:`U^{fk}`,  |                               +----------------------+-----------------------+
  | :math:`U^{ok}`   |                               | :math:`o_k\tn`       | ``(1)``               |
  +------------------+-------------------------------+----------------------+-----------------------+
  | :math:`V^{ik}`,  | ``(1, d_cell)``               | :math:`h\tn`         | ``(n_cell x d_cell)`` |
  | :math:`V^{fk}`,  |                               +----------------------+-----------------------+
  | :math:`V^{ok}`   |                               | :math:`z\tn`         | ``(d_emb)``           |
  +------------------+-------------------------------+----------------------+-----------------------+
  | :math:`b^{ik}`,  | ``(1)``                       | :math:`y\tn`         | ``(vocab_size)``      |
  | :math:`b^{fk}`,  |                               |                      |                       |
  | :math:`b^{ok}`   |                               |                      |                       |
  +------------------+-------------------------------+----------------------+-----------------------+
  | :math:`W^\ck`    | ``(d_cell, d_emb)``           |                                              |
  +------------------+-------------------------------+                                              |
  | :math:`U^\ck`    | ``(d_cell, n_cell x d_cell)`` |                                              |
  +------------------+-------------------------------+                                              |
  | :math:`b^\ck`    | ``(d_cell)``                  |                                              |
  +------------------+-------------------------------+                                              |
  | :math:`W^z`      | ``(d_emb, n_cell x d_cell)``  |                                              |
  +------------------+-------------------------------+                                              |
  | :math:`b^z`      | ``(d_emb)``                   |                                              |
  +------------------+-------------------------------+----------------------------------------------+

  - The differences between :py:class:`lmp.model.LSTM2000` and :py:class:`lmp.model.LSTM2002` are list as follow:

    - Input gate, forget gate and output gate units have peephole connections connect to memory cells' internal states.
      See :math:`\eqref{1}\eqref{2}\eqref{4}`.
    - The activation functions of memory cells' input and output are identity mappings instead of sigmoid functions.
      See :math:`\eqref{3}\eqref{5}`.
    - Output gate can only be calculated after updating memory cells' internal states.  See :math:`\eqref{4}`.

  Parameters
  ----------
  d_cell: int
    Memory cell dimension.
  d_emb: int
    Token embedding dimension.
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.
  n_cell: int
    Number of memory cells.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance.

  Attributes
  ----------
  c_0: torch.nn.Parameter
    Initial internal states of memory cells.
  d_cell: int
    Memory cell dimension.
  emb: torch.nn.Embedding
    Token embedding lookup table.
  h_0: torch.nn.Parameter
    Initial hidden states.
  loss_fn: torch.nn.CrossEntropyLoss
    Loss function to be optimized.
  model_name: ClassVar[str]
    CLI name of LSTM (2002 version) is ``LSTM-2002``.
  n_cell: int
    Number of memory cells.
  proj_e2c: torch.nn.Linear
    Fully connected layer which connects input units to memory cells.  Input dimension is ``d_emb``.  Output dimension
    is ``n_cell * (3 + d_cell)``.
  proj_h2c: torch.nn.Linear
    Fully connected layer which connects hidden states to memory cells.  Input dimension is ``n_cell * d_cell``.
    Output dimension is ``n_cell * (3 + d_cell)``.
  proj_h2e: torch.nn.Linear
    Fully connected layer which connects hidden states to embedding dimension.  Input dimension is ``n_cell * d_cell``.
    Output dimension is ``d_emb``.

  See Also
  --------
  lmp.model.BaseModel
    Language model utilities.
  lmp.model.ElmanNet
    LSTM (2002 version) language model.

  References
  ----------
  .. [1] Gers, F. A., Schraudolph, N. N., & Schmidhuber, J. (2002). `Learning precise timing with LSTM recurrent
         networks`_. Journal of machine learning research, 3(Aug), 115-143.

  .. _`Learning Precise Timing with LSTM Recurrent Networks`: https://www.jmlr.org/papers/v3/gers02a.html
  """

  model_name: ClassVar[str] = 'LSTM-2002'

  def __init__(self, *, d_cell: int, d_emb: int, n_cell: int, tknzr: BaseTknzr, **kwargs: Any):
    super().__init__(**kwargs)
    # `d_cell` validation.
    lmp.util.validate.raise_if_not_instance(val=d_cell, val_name='d_cell', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_cell], val_names=['1', 'd_cell'])
    self.d_cell = d_cell

    # `d_emb` validation.
    lmp.util.validate.raise_if_not_instance(val=d_emb, val_name='d_emb', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_emb], val_names=['1', 'd_emb'])

    # `n_cell` validation.
    lmp.util.validate.raise_if_not_instance(val=n_cell, val_name='n_cell', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_cell], val_names=['1', 'n_cell'])
    self.n_cell = n_cell

    # `tknzr` validation.
    lmp.util.validate.raise_if_not_instance(val=tknzr, val_name='tknzr', val_type=BaseTknzr)

    # Token embedding layer.  Use token ids to perform token embeddings lookup.
    self.emb = nn.Embedding(num_embeddings=tknzr.vocab_size, embedding_dim=d_emb, padding_idx=tknzr.pad_tkid)

    # Fully connected layer which connects input units to memory cells.
    self.proj_e2c = nn.Linear(in_features=d_emb, out_features=n_cell * (3 + d_cell))

    # Fully connected layer which connects hidden states to memory cells.
    self.proj_h2c = nn.Linear(in_features=n_cell * d_cell, out_features=n_cell * (3 + d_cell), bias=False)

    # Initial hidden states and initial memory cell internal states.  First dimension is set to `1` to broadcast along
    # batch dimension.
    self.h_0 = nn.Parameter(torch.zeros(1, n_cell * d_cell))
    self.c_0 = nn.Parameter(torch.zeros(1, n_cell, d_cell))

    # Peephole connections for gate units.  First dimension is set to `1` to broadcast along batch dimension.
    self.proj_c2ig = nn.Parameter(torch.zeros(1, n_cell, d_cell))
    self.proj_c2fg = nn.Parameter(torch.zeros(1, n_cell, d_cell))
    self.proj_c2og = nn.Parameter(torch.zeros(1, n_cell, d_cell))

    # Fully connected layer which project hidden states to embedding dimension.
    self.proj_h2e = nn.Linear(in_features=n_cell * d_cell, out_features=d_emb)

    # Calculate cross entropy loss for all non-padding tokens.
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=tknzr.pad_tkid)

    # Initialize model parameters.
    self.params_init()

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and non-gate units's biases are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\frac{-1}{\sqrt{v}}, \frac{1}{\sqrt{v}}}` where :math:`v =` ``max(d_emb, n_cell x d_cell)``.
    Input gate and output gate units' biases are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\frac{-1}{\sqrt{v}}, 0}`.  Forget gate units' biases are initialized with uniform
    distribution :math:`\mathcal{U}\pa{0, \frac{1}{\sqrt{v}}}`.

    Returns
    -------
    None
    """
    # Initialize weights and biases with uniform distribution.
    d_hid = self.n_cell * self.d_cell
    inv_sqrt_dim = 1 / math.sqrt(max(self.emb.embedding_dim, d_hid))
    nn.init.uniform_(self.emb.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_e2c.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_e2c.bias[:d_hid], -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_h2c.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.h_0, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.c_0, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_h2e.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_h2e.bias, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_c2ig, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_c2fg, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_c2og, -inv_sqrt_dim, inv_sqrt_dim)

    # Input gate and output gate units' biases are initialized to negative values.
    nn.init.uniform_(self.proj_e2c.bias[d_hid:d_hid + self.n_cell], -inv_sqrt_dim, 0.0)
    nn.init.uniform_(self.proj_e2c.bias[d_hid + 2 * self.n_cell:], -inv_sqrt_dim, 0.0)

    # Forget gate units' biases are initialized to positive values.
    nn.init.uniform_(self.proj_e2c.bias[d_hid + self.n_cell:d_hid + 2 * self.n_cell], 0.0, inv_sqrt_dim)

  def forward(self, batch_cur_tkids: torch.Tensor, batch_next_tkids: torch.Tensor) -> torch.Tensor:
    """Calculate language model training loss.

    This method must only be used to train model.  For inference use :py:meth:`lmp.model.ElmanNet.pred` instead.
    Forward pass algorithm is structured as follow:

    #. Use token ids to lookup token embeddings with ``self.emb``.
    #. Use ``self.proj_e2c`` and ``self.proj_h2c`` to calculate memory cells and gate units.  In this step we use
       teacher forcing, i.e., inputs are directly given instead generated by model.
    #. Use ``self.proj_h2e`` to project hidden states to embedding dimension.
    #. Calculate similarity scores by calculating inner product over all token embeddings.
    #. Return cross-entropy loss.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of token ids which represent input token ids of all time steps.  ``batch_cur_tkids`` has shape
      ``(batch_size, seq_len)`` and ``dtype == torch.int``.
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

    # Token embedding lookup and project from embedding layer to memory cells.
    # In  shape: (batch_size, seq_len).
    # Out shape: (batch_size, seq_len, n_cell x (3 + d_cell)).
    cells_and_gates_input_by_emb = self.proj_e2c(self.emb(batch_cur_tkids))

    # Perform recurrent calculation for `seq_len` steps.  We use teacher forcing, i.e., the current input `e[:, i, :]`
    # is used instead of generated by model.
    d_hid = self.n_cell * self.d_cell
    z_all = []
    c_prev: Union[torch.Tensor, nn.Parameter] = self.c_0
    h_prev: Union[torch.Tensor, nn.Parameter] = self.h_0
    for i in range(seq_len):
      # Project `h_prev` from hidden states to memory cells, then calculate memory cells and gates input activation.
      # shape: (batch_size, n_cell x (3 + d_cell)).
      cells_and_gates_common_input = cells_and_gates_input_by_emb[:, i, :] + self.proj_h2c(h_prev)

      # Get memory cells.
      # shape: (batch_size, n_cell, d_cell)
      cells_input = cells_and_gates_common_input[:, :d_hid].reshape(-1, self.n_cell, self.d_cell)

      # Calculate input gates and forget gates peephole connections.
      # shape: (batch_size, n_cell)
      input_gates_peephole_connection = (self.proj_c2ig * c_prev).sum(dim=2)
      forget_gates_peephole_connection = (self.proj_c2fg * c_prev).sum(dim=2)

      # Get input gates.
      # shape: (batch_size, n_cell, 1)
      input_gates = torch.sigmoid(
        cells_and_gates_common_input[:, d_hid:d_hid + self.n_cell] + input_gates_peephole_connection
      )
      input_gates = input_gates.unsqueeze(2)

      # Get forget gates.
      # shape: (batch_size, n_cell, 1)
      forget_gates = torch.sigmoid(
        cells_and_gates_common_input[:, d_hid + self.n_cell:d_hid + 2 * self.n_cell] + forget_gates_peephole_connection
      )
      forget_gates = forget_gates.unsqueeze(2)

      # Calculate current memory cells' internal states.
      # shape: (batch_size, n_cell, d_cell)
      c_cur = forget_gates * c_prev + input_gates * cells_input

      # Calculate output gates peephole connections.
      # shape: (batch_size, n_cell)
      output_gates_peephole_connection = (self.proj_c2og * c_cur).sum(dim=2)

      # Get output gates.
      # shape: (batch_size, n_cell, 1)
      output_gates = torch.sigmoid(
        cells_and_gates_common_input[:, d_hid + 2 * self.n_cell:] + output_gates_peephole_connection
      )
      output_gates = output_gates.unsqueeze(2)

      # Calculate current memory cells' outputs and reshape to fit the shape of hidden state.
      # shape: (batch_size, n_cell x d_cell)
      h_cur = output_gates * c_cur
      h_cur = h_cur.reshape(-1, d_hid)

      # Project from hidden states to embedding dimension.
      # shape: (batch_size, d_emb)
      z_all.append(torch.sigmoid(self.proj_h2e(h_cur)))

      # Update hidden states and memory cells' internal states.
      c_prev = c_cur
      h_prev = h_cur

    # Stack list of tensors into single tensor.
    # In  shape: list of (batch_size, d_emb) with length equals to `seq_len`.
    # Out shape: (batch_size, seq_len, d_emb).
    z = torch.stack(z_all, dim=1)

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # shape: (batch_size, seq_len, vocab_size).
    sim = z @ self.emb.weight.transpose(0, 1)

    # Reshape logits to calculate loss.
    # shape: (batch_size x seq_len, vocab_size).
    sim = sim.reshape(-1, self.emb.num_embeddings)

    # Reshape target to calculate loss.
    # shape: (batch_size x seq_len).
    batch_next_tkids = batch_next_tkids.reshape(-1)

    # Calculate cross-entropy loss.
    # Out shape : (1).
    return self.loss_fn(sim, batch_next_tkids)

  @torch.no_grad()
  def pred(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Calculate next token id probability distribution given previous hidden states and current input token id.

    This method must only be used for inference.  For training use :py:meth:`lmp.model.ElmanNet.forward` instead.  No
    tensor graphs will be constructed and no gradients will be calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.  ``batch_cur_tkids`` has shape ``(batch_size)`` and ``dtype == torch.int``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous calculation results.  Set to ``None`` to use ``[self.h_0, self.c_0]``.  ``batch_prev_states``
      must has two items, the first item will be used as hidden states and the second item will be used as memory
      cells' internal states.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the tensor of batch of next token id probability distribution with shape
      ``(batch_size, vocab_size)`` and ``dtype == torch.float``.  The second item in the tuple is a list of tensor
      which represent current hiddent states and current memory cells' internal states.
    """
    # Use initial hidden state if `batch_prev_state is None`.
    if batch_prev_states is None:
      batch_prev_states = [self.h_0, self.c_0]

    h_prev = batch_prev_states[0]
    c_prev = batch_prev_states[1]

    # Calculate memory cells and gate units common input, which consist of embeddings and previous hidden states.
    # In  shape: (batch_size).
    # Out shape: (batch_size, n_cell x (3 + d_cell)).
    cells_and_gates_common_input = self.proj_e2c(self.emb(batch_cur_tkids)) + self.proj_h2c(h_prev)

    # Calculate memory cells input activation and reshape to separate memory cells.
    # shape: (batch_size, n_cell, d_cell)
    d_hid = self.n_cell * self.d_cell
    cells_input = cells_and_gates_common_input[:, :d_hid].reshape(-1, self.n_cell, self.d_cell)

    # Calculate input gates and forget gates peephole connections.
    # shape: (batch_size, n_cell)
    input_gates_peephole_connection = (self.proj_c2ig * c_prev).sum(dim=2)
    forget_gates_peephole_connection = (self.proj_c2fg * c_prev).sum(dim=2)

    # Get input gates.
    # shape: (batch_size, n_cell, 1)
    input_gates = torch.sigmoid(
      cells_and_gates_common_input[:, d_hid:d_hid + self.n_cell] + input_gates_peephole_connection
    )
    input_gates = input_gates.unsqueeze(2)

    # Get forget gates.
    # shape: (batch_size, n_cell, 1)
    forget_gates = torch.sigmoid(
      cells_and_gates_common_input[:, d_hid + self.n_cell:d_hid + 2 * self.n_cell] + forget_gates_peephole_connection
    )
    forget_gates = forget_gates.unsqueeze(2)

    # Calculate current memory cells' internal states.
    # shape: (batch_size, n_cell, d_cell)
    c_cur = forget_gates * c_prev + input_gates * cells_input

    # Calculate output gates peephole connections.
    # shape: (batch_size, n_cell)
    output_gates_peephole_connection = (self.proj_c2og * c_cur).sum(dim=2)

    # Get output gates.
    # shape: (batch_size, n_cell, 1)
    output_gates = torch.sigmoid(
      cells_and_gates_common_input[:, d_hid + 2 * self.n_cell:] + output_gates_peephole_connection
    )
    output_gates = output_gates.unsqueeze(2)

    # Calculate current memory cells' outputs and reshape to fit the shape of hidden state.
    # shape: (batch_size, n_cell x d_cell)
    h_cur = output_gates * c_cur
    h_cur = h_cur.reshape(-1, d_hid)

    # Project from hidden states to embedding dimension.
    # shape: (batch_size, d_emb)
    z = torch.sigmoid(self.proj_h2e(h_cur))

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # shape: (batch_size, vocab_size).
    sim = z @ self.emb.weight.transpose(0, 1)

    # Calculate next token id probability distribution using softmax.
    # shape: (batch_size, vocab_size).
    batch_next_tkids_pd = F.softmax(sim, dim=-1)

    return (batch_next_tkids_pd, [h_cur, c_cur])

  @classmethod
  def train_parser(cls, parser: argparse.ArgumentParser) -> None:
    """CLI arguments parser for training LSTM (2002 version) language model.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI arguments parser.

    Returns
    -------
    None

    See Also
    --------
    lmp.model.BaseModel.train_parser
      CLI arguments parser for training language model.
    lmp.script.train_model
      Language model training script.

    Examples
    --------
    >>> import argparse
    >>> from lmp.model import LSTM2002
    >>> parser = argparse.ArgumentParser()
    >>> LSTM2002.train_parser(parser)
    >>> args = parser.parse_args([
    ...   '--batch_size', '32',
    ...   '--beta1', '0.9',
    ...   '--beta2', '0.99',
    ...   '--ckpt_step', '1000',
    ...   '--d_cell', '64',
    ...   '--d_emb', '100',
    ...   '--dset_name', 'wiki-text-2',
    ...   '--eps', '1e-8',
    ...   '--exp_name', 'my_exp',
    ...   '--log_step', '200',
    ...   '--lr', '1e-4',
    ...   '--max_norm', '1',
    ...   '--max_seq_len', '128',
    ...   '--n_cell', '8',
    ...   '--n_epoch', '10',
    ...   '--tknzr_exp_name', 'my_tknzr_exp',
    ...   '--ver', 'train',
    ...   '--wd', '1e-2',
    ... ])
    >>> args.batch_size == 32
    True
    >>> args.beta1 == 0.9
    True
    >>> args.beta2 == 0.99
    True
    >>> args.ckpt_step == 1000
    True
    >>> args.d_cell == 64
    True
    >>> args.d_emb == 100
    True
    >>> args.dset_name == 'wiki-text-2'
    True
    >>> args.eps == 1e-8
    True
    >>> args.exp_name == 'my_exp'
    True
    >>> args.log_step == 200
    True
    >>> args.lr == 1e-4
    True
    >>> args.max_norm == 1
    True
    >>> args.max_seq_len == 128
    True
    >>> args.n_cell == 8
    True
    >>> args.n_epoch == 10
    True
    >>> args.seed == 42
    True
    >>> args.tknzr_exp_name == 'my_tknzr_exp'
    True
    >>> args.ver == 'train'
    True
    >>> args.wd == 1e-2
    True
    """
    # Load common arguments.
    super().train_parser(parser=parser)

    # Required arguments.
    group = parser.add_argument_group('LSTM (2002 version) training arguments')
    group.add_argument(
      '--d_cell',
      help='Memory cell dimension.',
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
      '--n_cell',
      help='Number of memory cells.',
      required=True,
      type=int,
    )
