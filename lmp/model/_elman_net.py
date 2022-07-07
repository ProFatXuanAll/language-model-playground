"""Elman Net language model."""

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


class ElmanNet(BaseModel):
  r"""Elman Net [1]_ language model.

  Implement RNN model in the paper `Finding Structure in Time`_.

  - Let :math:`X = \set{x^0, x^1, \dots, x^{B-1}}` be a mini-batch of token id list.

    - The batch size of :math:`X` is :math:`B`.
    - All token id lists in :math:`X` have the same length :math:`S`.

  - Let :math:`V` be the vocabulary size of tokenizer.
  - Let :math:`x = (x[0], x[1], \dots, x[S-1])` be a token id list in :math:`X`.

    - For each :math:`t \in \set{0, \dots, S-1}`, the :math:`t`\-th token id in :math:`x` is defined as :math:`x[t]`.
    - Each token id is assigned with an unique token, i.e., :math:`x[t] \in \set{0, \dots, V -1}`.

  - Let :math:`\newcommand{\dEmb}{d_{\operatorname{emb}}} \dEmb` be the dimension of token embeddings.
  - Let :math:`\newcommand{\dHid}{d_{\operatorname{hid}}} \dHid` be the number of recurrent units.

  Elman Net is defined as follow.
  For each :math:`t \in \set{0, \dots, S-1}`, we input :math:`x[t]` and calculate the following terms:

  .. math::

    \newcommand{\pa}[1]{\left( #1 \right)}
    \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
    \begin{align*}
      e[t]   & = (x[t])\text{-th row of } E \text{ but treated as column vector}; \\
      h[t+1] & = \tanh(W^h \cdot e[t] + U^h \cdot h[t] + b^h);                    \\
      z[t+1] & = \tanh\pa{W^z \cdot h[t+1] + b^z};                                \\
      y[t+1] & = \sof{E \cdot z[t+1]}.
    \end{align*}

  +----------------------------------------+---------------------------------+
  | Trainable Parameters                   | Nodes                           |
  +--------------+-------------------------+--------------+------------------+
  | Parameter    | Shape                   | Symbol       | Shape            |
  +==============+=========================+==============+==================+
  | :math:`E`    | :math:`(V, \dEmb)`      | :math:`e[t]` | :math:`(\dEmb)`  |
  +--------------+-------------------------+--------------+------------------+
  | :math:`h[0]` | :math:`(\dHid)`         | :math:`h[t]` | :math:`(\dHid)`  |
  +--------------+-------------------------+--------------+------------------+
  | :math:`W^h`  | :math:`(\dHid, \dEmb)`  | :math:`z[t]` | :math:`(\dEmb)`  |
  +--------------+-------------------------+--------------+------------------+
  | :math:`U^h`  | :math:`(\dHid, \dHid)`  | :math:`y[t]` | :math:`(V)`      |
  +--------------+-------------------------+--------------+------------------+
  | :math:`b^h`  | :math:`(\dHid)`         |                                 |
  +--------------+-------------------------+                                 |
  | :math:`W^z`  | :math:`(\dEmb, \dHid)`  |                                 |
  +--------------+-------------------------+                                 |
  | :math:`b^z`  | :math:`(\dEmb)`         |                                 |
  +--------------+-------------------------+---------------------------------+

  - :math:`E` is the token embedding lookup table and :math:`e[t]` is the token embedding of :math:`x[t]`.
  - :math:`h[t]` represent the recurrent units in the Elman Net language model.
    The initial hidden state :math:`h[0]` is a trainable parameter.
  - :math:`z[t]` is obtained by transforming :math:`h[t]` from dimension :math:`\dHid` to :math:`\dEmb`.
  - The final output :math:`y[t]` is the next token id prediction probability distribution over tokenizer's vocabulary.
    We use inner product to calculate similarity scores over all token ids, and then use softmax to normalize
    similarity scores into probability range :math:`[0, 1]`.
  - Our implementation use :math:`\tanh` as activation function instead of the sigmoid function used in the paper.
    This is because optimal embeddings might have negative values.

  Parameters
  ----------
  d_emb: int
    Token embedding dimension :math:`\dEmb`.
  d_hid: int
    Hidden states dimension :math:`\dHid`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  p_emb: float
    Embeddings dropout probability.
  p_hid: float
    Hidden units dropout probability.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance.

  Attributes
  ----------
  emb: torch.nn.Embedding
    Token embedding lookup table :math:`E`.
    Input shape: :math:`(B, S)`.
    Output shape: :math:`(B, S, \dEmb)`.
  fc_e2h: torch.nn.Sequential
    Fully connected layer :math:`W^h` which connects input units to recurrent units.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \dHid)`.
  fc_h2e: torch.nn.Sequential
    Fully connected layer :math:`W^z` which transforms hidden states to next token embeddings.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dEmb)`.
  fc_h2h: torch.nn.Linear
    Fully connected layer :math:`U^h` which connects recurrent units to recurrent units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \dHid)`.
  h_0: torch.nn.Parameter
    Initial hidden states :math:`h[0]`.
    Shape: :math:`(1, \dHid)`
  model_name: ClassVar[str]
    CLI name of Elman Net is ``Elman-Net``.

  References
  ----------
  .. [1] Elman, J. L. (1990). `Finding Structure in Time`_. Cognitive science, 14(2), 179-211.

  .. _`Finding Structure in Time`: https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1
  """

  model_name: ClassVar[str] = 'Elman-Net'

  def __init__(
    self,
    *,
    d_emb: int,
    d_hid: int,
    p_emb: float,
    p_hid: float,
    tknzr: BaseTknzr,
    **kwargs: Any,
  ):
    super().__init__(**kwargs)

    # `d_emb` validation.
    lmp.util.validate.raise_if_not_instance(val=d_emb, val_name='d_emb', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_emb], val_names=['1', 'd_emb'])

    # `d_hid` validation.
    lmp.util.validate.raise_if_not_instance(val=d_hid, val_name='d_hid', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_hid], val_names=['1', 'd_hid'])

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

    # Fully connected layer which connects input units to recurrent units.
    # Dropout is applied to embeddings to make embeddings robust.
    self.fc_e2h = nn.Sequential(nn.Dropout(p=p_emb), nn.Linear(in_features=d_emb, out_features=d_hid))

    # Fully connected layer which connects recurrent units to recurrent units.
    # Set `bias=False` to share bias term with `self.fc_e2h` layer.
    # Do not apply dropout to hidden units since RNNs suffer from gradient vanish and dropout makes nodes (and thus
    # gradient) sparse.
    self.fc_h2h = nn.Linear(in_features=d_hid, out_features=d_hid, bias=False)

    # Initial hidden states.
    # First dimension is set to `1` to so that ``self.h_0`` can broadcast along batch dimension.
    self.h_0 = nn.Parameter(torch.zeros(1, d_hid))

    # Fully connected layer which transforms hidden states to next token embeddings.
    # Dropout is applied to make transform robust.
    self.fc_h2e = nn.Sequential(
      nn.Dropout(p=p_hid),
      nn.Linear(in_features=d_hid, out_features=d_emb),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{\dEmb}}, \dfrac{1}{\sqrt{\dEmb}}}`.

    Returns
    -------
    None
    """
    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(self.emb.embedding_dim)

    nn.init.uniform_(self.emb.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2h[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2h[1].bias, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2h.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.h_0, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2e[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2e[1].bias, -inv_sqrt_dim, inv_sqrt_dim)

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add Elman Net language model constructor parameters to CLI arguments parser.

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
    >>> from lmp.model import ElmanNet
    >>> parser = argparse.ArgumentParser()
    >>> ElmanNet.add_CLI_args(parser)
    >>> args = parser.parse_args([
    ...   '--d_emb', '2',
    ...   '--d_hid', '4',
    ...   '--p_emb', '0.5',
    ...   '--p_hid', '0.1',
    ... ])
    >>> assert args.d_emb == 2
    >>> assert args.d_hid == 4
    >>> assert math.isclose(args.p_emb, 0.5)
    >>> assert math.isclose(args.p_hid, 0.1)
    """
    # `parser` validation.
    lmp.util.validate.raise_if_not_instance(val=parser, val_name='parser', val_type=argparse.ArgumentParser)

    # Required arguments.
    group = parser.add_argument_group('Elman Net constructor arguments')
    group.add_argument(
      '--d_emb',
      help='Token embedding dimension.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--d_hid',
      help='Number of recurrent units.',
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

    Logits were calculated based on previous hidden states and current input token ids.
    Use :py:meth:`lmp.model.ElmanNet.pred` to convert logits into next token id probability distribution over
    tokenizer's vocabulary.
    Use :py:meth:`lmp.model.ElmanNet.loss` to convert logits into next token id prediction loss.
    Below we describe the forward pass algorithm of Elman Net language model.

    #. Use token ids to lookup token embeddings with ``self.emb``.
    #. Use ``self.fc_e2h`` and ``self.fc_h2h`` to calculate recurrent units.
       In this step we use teacher forcing, i.e., inputs are directly given instead of generated by model.
    #. Use ``self.fc_h2e`` to transform hidden states to next token embeddings.
    #. Perform inner product on token embeddings over tokenizer's vocabulary to get similarity scores.
    #. Return similarity scores (logits).

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states.
      There is only one tensor in the list, the shape of the tensor is :math:`(B, \dHid)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states :math:`h[0]`.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id logits with shape :math:`(B, S, V)` and
      ``dtype == torch.float``.
      The second item in the tuple is a single item list.
      The tensor in the list is the last hiddent states derived from current input token ids.
      The tensor has shape :math:`(B, \dHid)` and ``dtype == torch.float``.

    See Also
    --------
    lmp.tknzr.BaseTknzr.enc
      Source of token ids.
    """
    # Use initial hidden state if `batch_prev_state is None`.
    if batch_prev_states is None:
      batch_prev_states = [self.h_0]

    h_prev = batch_prev_states[0]

    # Sequence length.
    S = batch_cur_tkids.size(1)

    # Lookup token embeddings and feed to recurrent units.
    # In  shape: (B, S).
    # Out shape: (B, S, d_hid).
    e = self.fc_e2h(self.emb(batch_cur_tkids))

    # Perform recurrent calculation for `S` steps.
    h_all = []
    for t in range(S):
      # `e[:, t, :]` is the token embedding at time step `t`.
      # We use teacher forcing.
      # shape: (B, d_hid).
      # `h_prev` is the hidden states at time step `t`.
      # shape: (B, d_hid).
      # `h_cur` is the hidden states at time step `t + 1`.
      # shape: (B, d_hid).
      h_cur = torch.tanh(e[:, t, :] + self.fc_h2h(h_prev))

      h_all.append(h_cur)
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
    return (sim, [h_cur.detach()])

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
      Batch of previous hidden states.
      There is only one tensor in the list, the shape of the tensor is :math:`(B, \dHid)`.
      Set to ``None`` to use the initial hidden states :math:`h[0]`.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the mini-batch cross-entropy loss with shape :math:`(1)` and
      ``dtype == torch.float``.
      The second item in the tuple is a single item list.
      The tensor in the list is the last hiddent states derived from current input token ids.
      The tensor has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
    """
    # Get next token id logits and the last hidden states.
    # Logits shape: (B, S, V)
    # Last hidden states shape: [(B, d_hid)]
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate cross-entropy loss.
    # shape: (1).
    loss = lmp.util.metric.cross_entropy_loss(
      batch_tkids=batch_next_tkids,
      batch_tkids_pd=F.softmax(logits, dim=2),
    )

    # Return batch average loss.
    # shape: (1).
    return (loss, batch_cur_states)

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
      Batch of previous hidden states.
      There is only one tensor in the list, the shape of the tensor is :math:`(B, \dHid)`.
      Set to ``None`` to use the initial hidden states :math:`h[0]`.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id probability distribution over the tokenizer's
      vocabulary.
      Probability tensor has shape :math:`(B, S, V)` and ``dtype == torch.float``.
      The second item in the tuple is a single item list.
      The tensor in the list is the last hiddent states derived from current input token ids.
      The tensor has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
    """
    # Get next token id logits and the last hidden states.
    # Logits shape: (B, S, V)
    # Last hidden states shape: [(B, d_hid)]
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate next token id probability distribution using softmax.
    # shape: (B, S, V).
    return (F.softmax(logits, dim=-1), batch_cur_states)
