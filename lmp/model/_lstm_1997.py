"""LSTM (1997 version) language model."""

import argparse
import math
from typing import Any, ClassVar, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp.util.metric
import lmp.util.validate
from lmp.model._base import BaseModel
from lmp.tknzr._base import PAD_TKID, BaseTknzr


class LSTM1997(BaseModel):
  r"""LSTM (1997 version) [1]_ language model.

  Implement RNN model in the paper `Long Short-Term Memory`_.

  - Let :math:`X = \set{x^0, x^1, \dots, x^{B-1}}` be a mini-batch of token id list.

    - The batch size of :math:`X` is :math:`B`.
    - All token id lists in :math:`X` have the same length :math:`S`.

  - Let :math:`V` be the vocabulary size of tokenizer.
  - Let :math:`x = (x[0], x[1], \dots, x[S-1])` be a token id list in :math:`X`.

    - For each :math:`t \in \set{0, \dots, S-1}`, the :math:`t`\-th token id in :math:`x` is defined as :math:`x[t]`.
    - Each token id is assigned with an unique token, i.e., :math:`x[t] \in \set{0, \dots, V -1}`.

  - Let :math:`\newcommand{\dEmb}{d_{\operatorname{emb}}} \dEmb` be the dimension of token embeddings.
  - Let :math:`\newcommand{\nBlk}{n_{\operatorname{block}}} \nBlk` be the number of memory cell blocks.
  - Let :math:`\newcommand{\dBlk}{d_{\operatorname{block}}} \dBlk` be the dimension of each memory cell block.
  - Let :math:`\newcommand{\dHid}{d_{\operatorname{hid}}} \dHid = \nBlk \times \dBlk`.
  - Let :math:`\sigma` be the sigmoid function.

  LSTM (1997 version) is defined as follow.
  For each :math:`t \in \set{0, \dots, S-1}`, we input :math:`x[t]` and calculate the following terms:

  .. math::

     \newcommand{\pa}[1]{\left( #1 \right)}
     \newcommand{\c}{\operatorname{block}}
     \newcommand{\cn}[1]{{\c[#1]}}
     \newcommand{\ck}{{\cn{k}}}
     \newcommand{\hbar}{\overline{h}}
     \newcommand{\cat}[1]{\operatorname{concate}\pa{#1}}
     \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
     \begin{align*}
       e[t]           & = (x[t])\text{-th row of } E \text{ but treated as column vector};                     \\
       i[t]           & = \sigma\pa{W^i \cdot e[t] + U^i \cdot h[t] + b^i}                                     \\
       o[t]           & = \sigma\pa{W^o \cdot e[t] + U^o \cdot h[t] + b^o}                                     \\
       k              & \in \set{0, 1, \dots, \nBlk-1}                                                         \\
       g^\ck[t]       & = \tanh\pa{W^\ck \cdot e[t] + U^\ck \cdot h[t] + b^\ck}            && \tag{1}\label{1} \\
       c^\ck[t+1]     & = c^\ck[t] + i_k[t] \cdot g^\ck[t]                                                     \\
       \hbar^\ck[t+1] & = o_k[t] \cdot \tanh\pa{c^\ck[t+1]}                                && \tag{2}\label{2} \\
       h[t+1]         & = \cat{\hbar^\cn{0}[t+1], \dots, \hbar^\cn{\nBlk-1}[t+1]}                              \\
       z[t+1]         & = \tanh\pa{W^z \cdot h[t+1] + b^z}                                                     \\
       y[t+1]         & = \sof{E \cdot z[t+1]}
     \end{align*}

  +-------------------------------------------+----------------------------------------+
  | Trainable Parameters                      | Nodes                                  |
  +------------------+------------------------+----------------------+-----------------+
  | Parameter        | Shape                  | Symbol               | Shape           |
  +==================+========================+======================+=================+
  | :math:`E`        | :math:`(V, \dEmb)`     | :math:`e[t]`         | :math:`(\dEmb)` |
  +------------------+------------------------+----------------------+-----------------+
  | :math:`h[0]`     | :math:`(\dHid)`        | :math:`i[t]`,        | :math:`(\nBlk)` |
  |                  |                        | :math:`o[t]`         |                 |
  +------------------+------------------------+----------------------+-----------------+
  | :math:`W^i`,     | :math:`(\nBlk, \dEmb)` | :math:`i_k[t]`,      | :math:`(1)`     |
  | :math:`W^o`      |                        | :math:`o_k[t]`,      |                 |
  +------------------+------------------------+----------------------+-----------------+
  | :math:`U^i`,     | :math:`(\nBlk, \dHid)` | :math:`g^\ck[t]`,    | :math:`(\dBlk)` |
  | :math:`U^o`      |                        | :math:`c^\ck[t]`,    |                 |
  +------------------+------------------------+ :math:`\hbar^\ck[t]` |                 |
  | :math:`b^i`,     | :math:`(\nBlk)`        |                      |                 |
  | :math:`b^o`      |                        |                      |                 |
  +------------------+------------------------+----------------------+-----------------+
  | :math:`W^\ck`    | :math:`(\dBlk, \dEmb)` | :math:`h[t]`         | :math:`(\dHid)` |
  +------------------+------------------------+----------------------+-----------------+
  | :math:`U^\ck`    | :math:`(\dBlk, \dHid)` | :math:`z[t]`         | :math:`(\dEmb)` |
  +------------------+------------------------+----------------------+-----------------+
  | :math:`b^\ck`    | :math:`(\dBlk)`        | :math:`y[t]`         | :math:`(V)`     |
  +------------------+------------------------+----------------------+-----------------+
  | :math:`c^\ck[0]` | :math:`(\dBlk)`        |                                        |
  +------------------+------------------------+                                        |
  | :math:`W^z`      | :math:`(\dEmb, \dHid)` |                                        |
  +------------------+------------------------+                                        |
  | :math:`b^z`      | :math:`(\dEmb)`        |                                        |
  +------------------+------------------------+----------------------+-----------------+

  - :math:`E` is the token embedding lookup table defined as in :py:class:`lmp.model.ElmanNet`.
  - :math:`i[t], o[t]` are memory cell blocks' input and output gate units at time step :math:`t`, respectively.
    :math:`i_k[t], o_k[t]` are their :math:`k`-th coordinates, respectively.
  - There are :math:`\nBlk` different memory cell blocks :math:`\cn{0}, \dots, \cn{\nBlk-1}`.
    For each :math:`k \in \set{0, \dots, \nBlk-1}`, the :math:`k`-th memory cell block have the following components:

    - Input activation :math:`g^\ck[t]`.
    - Input gate unit :math:`i_k[t]`.
    - Output gate unit :math:`o_k[t]`.
    - Internal states :math:`c^\ck[t]`.
      The initial internal states :math:`c^\ck[0]` is a trainable parameter.
    - Output :math:`\hbar^\ck[t]`.

  - The hidden states :math:`h[t]` at time step :math:`t` is the concatenation of all memory cell blocks' outputs.
    The initial hidden states :math:`h[0]` is a trainable parameter.
  - The calculations after hidden states are the same as :py:class:`lmp.model.ElmanNet`.
  - Our implementation use :math:`\tanh` as memory cell blocks' input activation function.
    The implementation in the paper use :math:`4 \sigma - 2` in :math:`\eqref{1}` and :math:`2 \sigma - 1` in
    :math:`\eqref{2}`.
    We argue that the change in :math:`\eqref{1}` does not greatly affect the computation result and :math:`\eqref{2}`
    is the same as the paper implementation
    (to be precise, when using :math:`\tanh`, the gradient will be scaled by 2 compare to sigmoid function).

  Parameters
  ----------
  d_blk: int
    Dimension of each memory cell block :math:`\dBlk`.
  d_emb: int
    Token embedding dimension :math:`\dEmb`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.
  p_emb: float
    Embeddings dropout probability.
  p_hid: float
    Hidden units dropout probability.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance.

  Attributes
  ----------
  c_0: torch.nn.Parameter
    Initial internal states of memory cell blocks :math:`\pa{c^\cn{0}[0], \dots, c^\cn{\nBlk-1}[0]}`.
    Shape: :math:`(1, \nBlk, \dBlk)`.
  d_blk: int
    Dimension of each memory cell block :math:`\dBlk`.
  d_hid: int
    Total number of memory cell units :math:`\dHid`.
  emb: torch.nn.Embedding
    Token embedding lookup table :math:`E`.
    Input shape: :math:`(B, S)`.
    Output shape: :math:`(B, S, \dEmb)`.
  fc_e2ig: torch.nn.Sequential
    Fully connected layer :math:`W^i` which connects input units to memory cell's input gate units.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \nBlk)`.
  fc_e2mc_in: torch.nn.Sequential
    Fully connected layer :math:`\pa{W^\cn{0}, \dots, W^\cn{\nBlk-1}}` which connects input units to memory cell
    blocks' input activations.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \dHid)`.
  fc_e2og: torch.nn.Sequential
    Fully connected layer :math:`W^o` which connects input units to memory cell's output gate units.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \nBlk)`.
  fc_h2e: torch.nn.Sequential
    Fully connected layer :math:`W^z` which transforms hidden states to next token embeddings.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dEmb)`.
  fc_h2ig: torch.nn.Linear
    Fully connected layer :math:`U^i` which connects hidden states to memory cell's input gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  fc_h2mc_in: torch.nn.Linear
    Fully connected layer :math:`\pa{U^\cn{0}, \dots, U^\cn{\nBlk-1}}` which connects hidden states to memory cell
    blocks' input activations.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \dHid)`.
  fc_h2og: torch.nn.Linear
    Fully connected layer :math:`U^o` which connects hidden states to memory cell's output gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  h_0: torch.nn.Parameter
    Initial hidden states :math:`h[0]`.
    Shape: :math:`(1, \dHid)`
  model_name: ClassVar[str]
    CLI name of LSTM (1997 version) is ``LSTM-1997``.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.

  See Also
  --------
  :doc:`lmp.model.BaseModel </model/BaseModel>`
    Language model utilities.

  References
  ----------
  .. [1] S. Hochreiter and J. Schmidhuber, "`Long Short-Term Memory`_," in Neural Computation, vol. 9, no. 8,
     pp. 1735-1780, 15 Nov. 1997, doi: 10.1162/neco.1997.9.8.1735.

  .. _`Long Short-Term Memory`: https://ieeexplore.ieee.org/abstract/document/6795963
  """

  model_name: ClassVar[str] = 'LSTM-1997'

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

    # Token embedding layer.
    # Use token ids to perform token embeddings lookup.
    self.emb = nn.Embedding(num_embeddings=tknzr.vocab_size, embedding_dim=d_emb, padding_idx=PAD_TKID)

    # Fully connected layer which connects input units to input / output gate units.
    # Dropout is applied to embeddings to make embeddings robust.
    self.fc_e2ig = nn.Sequential(nn.Dropout(p=p_emb), nn.Linear(in_features=d_emb, out_features=n_blk))
    self.fc_e2og = nn.Sequential(nn.Dropout(p=p_emb), nn.Linear(in_features=d_emb, out_features=n_blk))

    # Fully connected layer which connects hidden states to input / output gate units.
    # Set `bias=False` to share bias term with `self.fc_e2ig` and `self.fc_e2og` layers.
    self.fc_h2ig = nn.Linear(in_features=self.d_hid, out_features=n_blk, bias=False)
    self.fc_h2og = nn.Linear(in_features=self.d_hid, out_features=n_blk, bias=False)

    # Fully connected layer which connects input units to memory cell blocks' input activation.
    # Dropout is applied to embeddings to make embeddings robust.
    self.fc_e2mc_in = nn.Sequential(nn.Dropout(p=p_emb), nn.Linear(in_features=d_emb, out_features=self.d_hid))

    # Fully connected layer which connects hidden states to memory cell blocks' input activation.
    # Set `bias=False` to share bias term with `self.fc_e2mc_in` layer.
    self.fc_h2mc_in = nn.Linear(in_features=self.d_hid, out_features=self.d_hid, bias=False)

    # Initial hidden states and initial memory cell internal states.
    # First dimension is set to `1` to so that they can broadcast along batch dimension.
    self.h_0 = nn.Parameter(torch.zeros(1, self.d_hid))
    self.c_0 = nn.Parameter(torch.zeros(1, n_blk, d_blk))

    # Fully connected layer which transforms hidden states to next token embeddings.
    # Dropout is applied to make transform robust.
    self.fc_h2e = nn.Sequential(
      nn.Dropout(p=p_hid),
      nn.Linear(in_features=self.d_hid, out_features=d_emb),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases other than :math:`b^i, b^o` are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, \dfrac{1}{\sqrt{d}}}` where :math:`d = \max(\dEmb, \dHid)`.
    :math:`b^i, b^o` are initialized with uniform distribution :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, 0}` so that
    input and output gates remain closed at the begining of training.

    Returns
    -------
    None
    """
    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(max(self.emb.embedding_dim, self.d_hid))

    nn.init.uniform_(self.emb.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2ig[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2og[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2mc_in[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2mc_in[1].bias, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2ig.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2og.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2mc_in.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.h_0, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.c_0, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2e[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2e[1].bias, -inv_sqrt_dim, inv_sqrt_dim)

    # Gate units' biases are initialized to negative values.
    nn.init.uniform_(self.fc_e2ig[1].bias, -inv_sqrt_dim, 0.0)
    nn.init.uniform_(self.fc_e2og[1].bias, -inv_sqrt_dim, 0.0)

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """CLI arguments parser for training LSTM (1997 version) language model.

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
    >>> from lmp.model import LSTM1997
    >>> parser = argparse.ArgumentParser()
    >>> LSTM1997.add_CLI_args(parser)
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
    group = parser.add_argument_group('LSTM (1997 version) constructor arguments')
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

  def forward(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    r"""Calculate next token id logits.

    Logits were calculated based on previous hidden states, previous internal cell states and and current input token
    ids.
    Use :py:meth:`lmp.model.LSTM1997.pred` to convert logits into next token id probability distribution over
    tokenizer's vocabulary.
    Use :py:meth:`lmp.model.LSTM1997.loss` to convert logits into next token id prediction loss.
    Below we describe the forward pass algorithm of LSTM (1997 version) language model.

    #. Use token ids to lookup token embeddings with ``self.emb``.
    #. Use ``self.fc_e2ig`` and ``self.fc_h2ig`` to calculate memory cell blocks' input gate units.
    #. Use ``self.fc_e2og`` and ``self.fc_h2og`` to calculate memory cell blocks' output gate units.
    #. Use ``self.fc_e2mc_in`` and ``self.fc_h2mc_in`` to calculate memory cell blocks' input activations.
    #. Update memory cell block's internal states.
    #. Calculate memory cell block's output.
    #. Update hidden states.
    #. Use ``self.fc_h2e`` to transform hidden states to next token embeddings.
    #. Perform inner product on token embeddings over tokenizer's vocabulary to get similarity scores.
    #. Return similarity scores (logits).

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states and internal states.
      There are two tensors in the list.
      The first tensor is batch of previous hidden states with shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      The second tensor is batch of previous internal states with shape :math:`(B, \nBlk, \dBlk)` and
      ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states :math:`h[0]` and initial internal states :math:`c[0]`.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id logits with shape :math:`(B, S, V)` and
      ``dtype == torch.float``.
      The second item in the tuple is a two items list.
      The first tensor in the list is the last hiddent states derived from current input token ids.
      The first tensor has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      The second tensor in the list is the last internal states derived from current input token ids.
      The second tensor has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
    """
    # Use initial hidden states if `batch_prev_state is None`.
    if batch_prev_states is None:
      batch_prev_states = [self.h_0, self.c_0]

    h_prev = batch_prev_states[0]
    c_prev = batch_prev_states[1]

    # Sequence length.
    S = batch_cur_tkids.size(1)

    # Lookup token embeddings.
    # In  shape: (B, S).
    # Out shape: (B, S, d_emb).
    e = self.emb(batch_cur_tkids)

    # Feed token embeddings to input / output gate units.
    # In  shape: (B, S).
    # Out shape: (B, S, n_blk).
    e2ig = self.fc_e2ig(e)
    e2og = self.fc_e2og(e)

    # Feed token embeddings to memory cell blocks.
    # In  shape: (B, S).
    # Out shape: (B, S, d_hid).
    e2mc_in = self.fc_e2mc_in(e)

    # Perform recurrent calculation for `S` steps.
    # We use teacher forcing, i.e., the current input is used instead of generated by model.
    h_all = []
    for t in range(S):
      # Get input gate and output gate units and unsqueeze to separate memory cell blocks.
      # shape: (B, n_blk, 1).
      ig = torch.sigmoid(e2ig[:, t, :] + self.fc_h2ig(h_prev)).unsqueeze(-1)
      og = torch.sigmoid(e2og[:, t, :] + self.fc_h2og(h_prev)).unsqueeze(-1)

      # Calculate memory cell blocks input activation and reshape to separate memory cell blocks.
      # shape: (B, n_blk, d_blk).
      mc_in = torch.tanh(e2mc_in[:, t, :] + self.fc_h2mc_in(h_prev)).reshape(-1, self.n_blk, self.d_blk)

      # Calculate memory cell blocks' new internal states.
      # shape: (B, n_blk, d_blk).
      c_cur = c_prev + ig * mc_in

      # Calculate memory cell blocks' outputs and concatenate them to form the new hidden states.
      # shape: (B, d_hid).
      h_cur = (og * torch.tanh(c_cur)).reshape(-1, self.d_hid)

      h_all.append(h_cur)

      # Update hidden states and memory cell blocks' internal states.
      c_prev = c_cur
      h_prev = h_cur

    # Stack list of tensors into single tensor.
    # In  shape: list of (B, d_hid) with length equals to `S`.
    # Out shape: (B, S, d_hid).
    h = torch.stack(h_all, dim=1)

    # Transform hidden states to next token embeddings.
    # shape: (B, S, d_emb).
    z = self.fc_h2e(h)

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # shape: (B, S, V).
    sim = z @ self.emb.weight.transpose(0, 1)
    return (sim, [h_cur.detach(), c_cur.detach()])

  def loss(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    r"""Calculate language model prediction loss.

    Loss is estimated by cross entropy.
    This method must only be used for training.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_next_tkids: torch.Tensor
      Ground truth of each sample in the batch.
      ``batch_next_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states and internal states.
      There are two tensors in the list.
      The first tensor is batch of previous hidden states with shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      The second tensor is batch of previous internal states with shape :math:`(B, \nBlk, \dBlk)` and
      ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states :math:`h[0]` and initial internal states :math:`c[0]`.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the mini-batch cross-entropy loss with shape :math:`(1)` and
      ``dtype == torch.float``.
      The second item in the tuple is a two items list.
      The first tensor in the list is the last hiddent states derived from current input token ids.
      The first tensor has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      The second tensor in the list is the last internal states derived from current input token ids.
      The second tensor has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
    """
    # Get next token id logits and the last hidden states.
    # Logits shape: (B, S, V)
    # Last hidden states and internal states shapes: [(B, d_hid), (B, n_blk, d_blk)]
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate cross-entropy loss.
    # shape: (B).
    loss = lmp.util.metric.cross_entropy_loss(
      batch_tkids=batch_next_tkids,
      batch_tkids_pd=F.softmax(logits, dim=2),
    )

    # Return batch average loss.
    # shape: (1).
    return (loss.mean(), batch_cur_states)

  @torch.no_grad()
  def pred(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    r"""Calculate next token id probability distribution over tokenizer's vocabulary.

    Probabilities were calculated based on previous hidden states and current input token id.
    This method must only be used for inference.
    No tensor graphs will be constructed and no gradients will be calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states and internal states.
      There are two tensors in the list.
      The first tensor is batch of previous hidden states with shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      The second tensor is batch of previous internal states with shape :math:`(B, \nBlk, \dBlk)` and
      ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states :math:`h[0]` and initial internal states :math:`c[0]`.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id probability distribution over the tokenizer's
      vocabulary.
      Probability tensor has shape :math:`(B, S, V)` and ``dtype == torch.float``.
      The second item in the tuple is a two items list.
      The first tensor in the list is the last hiddent states derived from current input token ids.
      The first tensor has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      The second tensor in the list is the last internal states derived from current input token ids.
      The second tensor has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
    """
    # Get next token id logits and the last hidden states.
    # Logits shape: (B, S, V)
    # Last hidden states and internal states shapes: [(B, d_hid), (B, n_blk, d_blk)]
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate next token id probability distribution using softmax.
    # shape: (B, S, V).
    return (F.softmax(logits, dim=-1), batch_cur_states)
