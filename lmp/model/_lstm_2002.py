"""LSTM (2002 version) language model."""

import argparse
import math
from typing import Any, ClassVar, List, Optional, Tuple

import torch
import torch.nn as nn

import lmp.util.metric
import lmp.util.validate
from lmp.model._lstm_2000 import LSTM2000
from lmp.tknzr._base import BaseTknzr


class LSTM2002(LSTM2000):
  r"""LSTM (2002 version) [1]_ language model.

  Implement RNN model in the paper `Learning Precise Timing with LSTM Recurrent Networks`_.

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

  LSTM (2002 version) is defined as follow.
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
       e[t]           & = (x[t])\text{-th row of } E \text{ but treated as column vector};                        \\
       k              & \in \set{0, 1, \dots, \nBlk - 1}                                                          \\
       i_k[t]         & = \sigma\pa{W^{i_k} \odot e[t] + U^{i_k} \odot h[t] + V^{i_k} \odot c^\ck[t] + b^{i_k}}
                      &&  \tag{1}\label{1}                                                                        \\
       f_k[t]         & = \sigma\pa{W^{f_k} \odot e[t] + U^{f_k} \odot h[t] + V^{f_k} \odot c^\ck[t] + b^{f_k}}
                      && \tag{2}\label{2}                                                                         \\
       g^\ck[t]       & = \tanh\pa{W^\ck \cdot e[t] + U^\ck \cdot h[t] + b^\ck}
                      && \tag{3}\label{3}                                                                         \\
       c^\ck[t+1]     & = f_k[t] \cdot c^\ck[t] + i_k[t] \cdot g^\ck[t]                                           \\
       o_k[t+1]       & = \sigma\pa{W^{o_k} \odot e[t] + U^{o_k} \odot h[t] + V^{o_k} \odot c^\ck[t+1] + b^{o_k}}
                      && \tag{4}\label{4}                                                                         \\
       \hbar^\ck[t+1] & = o_k[t+1] \cdot \tanh\pa{c^\ck[t+1]}
                      && \tag{5}\label{5}                                                                         \\
       h[t+1]         & = \cat{\hbar^\cn{0}[t+1], \dots, \hbar^\cn{\nBlk-1}[t+1]}                                 \\
       z[t+1]         & = \tanh\pa{W^z \cdot h[t+1] + b^z}                                                        \\
       y[t+1]         & = \sof{E \cdot z[t+1]}
     \end{align*}

  +-------------------------------------------+-------------------------------------+
  | Trainable Parameters                      | Nodes                               |
  +------------------+------------------------+-------------------+-----------------+
  | Parameter        | Shape                  | Symbol            | Shape           |
  +==================+========================+===================+=================+
  | :math:`E`        | :math:`(V, \dEmb)`     | :math:`e[t]`      | :math:`(\dEmb)` |
  +------------------+------------------------+-------------------+-----------------+
  | :math:`h[0]`     | :math:`(\dHid)`        | :math:`i_k[t]`,   | :math:`(1)`     |
  |                  |                        | :math:`f_k[t]`,   |                 |
  +                  |                        | :math:`o_k[t]`    |                 |
  +------------------+------------------------+-------------------+-----------------+
  | :math:`c^\ck[0]` | :math:`(\dBlk)`        | :math:`g^\ck[t]`, | :math:`(\dBlk)` |
  |                  |                        | :math:`c^\ck[t]`  |                 |
  +------------------+------------------------+-------------------+-----------------+
  | :math:`W^{i_k}`, | :math:`(\dEmb)`        | :math:`h[t]`      | :math:`(\dHid)` |
  | :math:`W^{f_k}`, |                        +-------------------+-----------------+
  | :math:`W^{o_k}`  |                        | :math:`z[t]`      | :math:`(\dEmb)` |
  +------------------+------------------------+-------------------+-----------------+
  | :math:`U^{i_k}`, | :math:`(\dHid)`        | :math:`y[t]`      | :math:`(V)`     |
  | :math:`U^{f_k}`, |                        |                   |                 |
  | :math:`U^{o_k}`  |                        |                   |                 |
  +------------------+------------------------+-------------------+-----------------+
  | :math:`V^{i_k}`, | :math:`(\dBlk)`        |                                     |
  | :math:`V^{f_k}`, |                        |                                     |
  | :math:`V^{o_k}`  |                        |                                     |
  +------------------+------------------------+                                     |
  | :math:`b^{i_k}`, | :math:`(1)`            |                                     |
  | :math:`b^{f_k}`, |                        |                                     |
  | :math:`b^{o_k}`  |                        |                                     |
  +------------------+------------------------+                                     |
  | :math:`W^\ck`    | :math:`(\dBlk, \dEmb)` |                                     |
  +------------------+------------------------+                                     |
  | :math:`U^\ck`    | :math:`(\dBlk, \dHid)` |                                     |
  +------------------+------------------------+                                     |
  | :math:`b^\ck`    | :math:`(\dBlk)`        |                                     |
  +------------------+------------------------+                                     |
  | :math:`W^z`      | :math:`(\dEmb, \dHid)` |                                     |
  +------------------+------------------------+                                     |
  | :math:`b^z`      | :math:`(\dEmb)`        |                                     |
  +------------------+------------------------+-------------------------------------+

  - The differences between :py:class:`lmp.model.LSTM2000` and :py:class:`lmp.model.LSTM2002` are list as follow:

    - Input, forget and output gate units have peephole connections connect to memory cell blocks' internal states.
      See :math:`\eqref{1}\eqref{2}\eqref{4}`.
    - Output gate units can only be calculated after updating memory cell blocks' internal states.
      See :math:`\eqref{4}`.

  - Our implementation use :math:`\tanh` as activation function instead of identity mapping.
    The implementation in the paper use identity mappings in :math:`\eqref{3}\eqref{5}`.
    We argue that the changes in :math:`\eqref{3}\eqref{5}` make sure our model activations are bounded.
    The implementation in the paper is unbounded.
    Since one usually use much larger dimension to train language model compare to the paper (which use dimension
    :math:`1` on everything), activations of LSTM tend to grow to extremely positive / negative values without
    :math:`\tanh`.

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
  fc_e2fg: torch.nn.Sequential
    Fully connected layer :math:`\pa{W^{f_0}, \dots, W^{f_{\nBlk-1}}}` which connects input units to memory cell's
    forget gate units.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \nBlk)`.
  fc_e2ig: torch.nn.Sequential
    Fully connected layer :math:`\pa{W^{i_0}, \dots, W^{i_{\nBlk-1}}}` which connects input units to memory cell's
    input gate units.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \nBlk)`.
  fc_e2mc_in: torch.nn.Sequential
    Fully connected layer :math:`\pa{W^\cn{0}, \dots, W^\cn{\nBlk-1}}` which connects input units to memory cell
    blocks' input activations.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \dHid)`.
  fc_e2og: torch.nn.Sequential
    Fully connected layer :math:`\pa{W^{o_0}, \dots, W^{o_{\nBlk-1}}}` which connects input units to memory cell's
    output gate units.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \nBlk)`.
  fc_h2e: torch.nn.Sequential
    Fully connected layer :math:`W^z` which transforms hidden states to next token embeddings.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dEmb)`.
  fc_h2fg: torch.nn.Linear
    Fully connected layer :math:`\pa{U^{f_0}, \dots, U^{f_{\nBlk-1}}}` which connects hidden states to memory cell's
    forget gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  fc_h2ig: torch.nn.Linear
    Fully connected layer :math:`\pa{U^{i_0}, \dots, U^{i_{\nBlk-1}}}` which connects hidden states to memory cell's
    input gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  fc_h2mc_in: torch.nn.Linear
    Fully connected layer :math:`\pa{U^\cn{0}, \dots, U^\cn{\nBlk-1}}` which connects hidden states to memory cell
    blocks' input activations.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \dHid)`.
  fc_h2og: torch.nn.Linear
    Fully connected layer :math:`\pa{U^{o_0}, \dots, U^{o_{\nBlk-1}}}` which connects hidden states to memory cell's
    output gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  h_0: torch.nn.Parameter
    Initial hidden states :math:`h[0]`.
    Shape: :math:`(1, \dHid)`
  model_name: ClassVar[str]
    CLI name of LSTM (2002 version) is ``LSTM-2002``.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.
  pc_c2fg: torch.nn.Parameter
    Peephole connections :math:`\pa{V^{f_0}, \dots, V^{f_{\nBlk-1}}}` which connect the :math:`k`-th memory cell
    blocks' internal states to the :math:`k`-th forget gate units.
    Shape: :math:`(1, \nBlk, \dBlk)`.
  pc_c2ig: torch.nn.Parameter
    Peephole connections :math:`\pa{V^{i_0}, \dots, V^{i_{\nBlk-1}}}` which connect the :math:`k`-th memory cell
    blocks' internal states to the :math:`k`-th input gate units.
    Shape: :math:`(1, \nBlk, \dBlk)`.
  pc_c2og: torch.nn.Parameter
    Peephole connections :math:`\pa{V^{o_0}, \dots, V^{o_{\nBlk-1}}}` which connect the :math:`k`-th memory cell
    blocks' internal states to the :math:`k`-th output gate units.
    Shape: :math:`(1, \nBlk, \dBlk)`.

  See Also
  --------
  :doc:`lmp.model.LSTM2000 </model/LSTM2000>`
    LSTM (2000 version) language model.

  References
  ----------
  .. [1] Gers, F. A., Schraudolph, N. N., & Schmidhuber, J. (2002). `Learning precise timing with LSTM recurrent
         networks`_. Journal of machine learning research, 3(Aug), 115-143.

  .. _`Learning Precise Timing with LSTM Recurrent Networks`: https://www.jmlr.org/papers/v3/gers02a.html
  """

  model_name: ClassVar[str] = 'LSTM-2002'

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
    super().__init__(
      d_blk=d_blk,
      d_emb=d_emb,
      n_blk=n_blk,
      p_emb=p_emb,
      p_hid=p_hid,
      tknzr=tknzr,
      **kwargs,
    )

    # Peephole connections for gate units.
    # First dimension is set to `1` to broadcast along batch dimension.
    self.pc_c2fg = nn.Parameter(torch.zeros(1, n_blk, d_blk))
    self.pc_c2ig = nn.Parameter(torch.zeros(1, n_blk, d_blk))
    self.pc_c2og = nn.Parameter(torch.zeros(1, n_blk, d_blk))

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases other than :math:`b^f, b^i, b^o` are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, \dfrac{1}{\sqrt{d}}}` where :math:`d = \max(\dEmb, \dHid)`.
    :math:`b^i, b^o` are initialized with uniform distribution :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, 0}` so that
    input and output gates remain closed at the begining of training.
    :math:`b^f` are initialized with uniform distribution :math:`\mathcal{U}\pa{0, \dfrac{1}{\sqrt{d}}}` so that forget
    gates remain open at the begining of training.

    Returns
    -------
    None
    """
    super().params_init()

    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(max(self.emb.embedding_dim, self.d_hid))

    nn.init.uniform_(self.pc_c2fg, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.pc_c2ig, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.pc_c2og, -inv_sqrt_dim, inv_sqrt_dim)

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
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
    :doc:`lmp.script.train_model </script/train_model>`
      Language model training script.

    Examples
    --------
    >>> import argparse
    >>> import math
    >>> from lmp.model import LSTM2002
    >>> parser = argparse.ArgumentParser()
    >>> LSTM2002.add_CLI_args(parser)
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
    group = parser.add_argument_group('LSTM (2002 version) constructor arguments')
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
    Use :py:meth:`lmp.model.LSTM2002.pred` to convert logits into next token id probability distribution over
    tokenizer's vocabulary.
    Use :py:meth:`lmp.model.LSTM2002.loss` to convert logits into next token id prediction loss.
    Below we describe the forward pass algorithm of LSTM (2002 version) language model.

    #. Use token ids to lookup token embeddings with ``self.emb``.
    #. Use ``self.fc_e2fg, self.fc_h2fg, self.pc_c2fg`` to calculate memory cell blocks' forget gate units.
    #. Use ``self.fc_e2ig, self.fc_h2ig, self.pc_c2ig`` to calculate memory cell blocks' input gate units.
    #. Use ``self.fc_e2mc_in`` and ``self.fc_h2mc_in`` to calculate memory cell blocks' input activations.
    #. Update memory cell block's internal states.
    #. Use ``self.fc_e2og, self.fc_h2og, self.pc_c2og`` and new internal states to calculate memory cell blocks' output
       gate units.
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

    # Feed token embeddings to forget / input / output gate units.
    # In  shape: (B, S).
    # Out shape: (B, S, n_blk).
    e2fg = self.fc_e2fg(e)
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
      # Calculate forget gate and input gate units peephole connections.
      # shape: (B, n_blk).
      c2fg = (self.pc_c2fg * c_prev).sum(dim=-1)
      c2ig = (self.pc_c2ig * c_prev).sum(dim=-1)

      # Get forget gate and input gate units and unsqueeze to separate memory cell blocks.
      # shape: (B, n_blk, 1).
      fg = torch.sigmoid(e2fg[:, t, :] + self.fc_h2fg(h_prev) + c2fg).unsqueeze(-1)
      ig = torch.sigmoid(e2ig[:, t, :] + self.fc_h2ig(h_prev) + c2ig).unsqueeze(-1)

      # Calculate memory cell blocks input activation and reshape to separate memory cell blocks.
      # shape: (B, n_blk, d_blk).
      mc_in = torch.tanh(e2mc_in[:, t, :] + self.fc_h2mc_in(h_prev)).reshape(-1, self.n_blk, self.d_blk)

      # Calculate memory cell blocks' new internal states.
      # shape: (B, n_blk, d_blk).
      c_cur = fg * c_prev + ig * mc_in

      # Calculate output gate units peephole connections.
      # shape: (B, n_blk).
      c2og = (self.pc_c2og * c_cur).sum(dim=-1)

      # Get output gate units and unsqueeze to separate memory cell blocks.
      # shape: (B, n_blk, 1).
      og = torch.sigmoid(e2og[:, t, :] + self.fc_h2og(h_prev) + c2og).unsqueeze(-1)

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
