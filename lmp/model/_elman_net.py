"""Elman Net language model."""

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


class ElmanNet(BaseModel):
  r"""Elman Net [1]_ language model.

  Implement RNN model in the paper `Finding Structure in Time`_.

  - Let :math:`x` be the input token id list as defined in :py:class:`lmp.model.BaseModel`.
  - Let ``d_emb`` be the dimension of token embeddings and let ``vocab_size`` be the vocabulary size of tokenizer.
  - Let ``d_hid`` be the number of recurrent units.

  Elman Net is defined as follow:

  .. math::

     \newcommand{\pa}[1]{\left( #1 \right)}
     \newcommand{\set}[1]{\left\lbrace #1 \right\rbrace}
     \newcommand{\t}{[t]}
     \newcommand{\tp}{[t - 1]}
     \newcommand{\tz}{[0]}
     \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
     \begin{align*}
       e\t & = (x\t)\text{-th column of } E                   \\
       h\t & = \tanh\pa{W^h \cdot e\t + U^h \cdot h\tp + b^h} \\
       z\t & = \tanh\pa{W^z \cdot h\t + b^z}                  \\
       y\t & = \sof{E^{\top} \cdot z\t}
     \end{align*}

  +----------------------------------------+--------------------------------+
  | Trainable Parameters                   | Nodes                          |
  +--------------+-------------------------+-------------+------------------+
  | Parameter    | Shape                   | Symbol      | Shape            |
  +==============+=========================+=============+==================+
  | :math:`E`    | ``(d_emb, vocab_size)`` | :math:`e\t` | ``(d_emb)``      |
  +--------------+-------------------------+-------------+------------------+
  | :math:`h\tz` | ``(d_hid)``             | :math:`h\t` | ``(d_hid)``      |
  +--------------+-------------------------+-------------+------------------+
  | :math:`W^h`  | ``(d_hid, d_emb)``      | :math:`z\t` | ``(d_emb)``      |
  +--------------+-------------------------+-------------+------------------+
  | :math:`U^h`  | ``(d_hid, d_hid)``      | :math:`y\t` | ``(vocab_size)`` |
  +--------------+-------------------------+-------------+------------------+
  | :math:`b^h`  | ``(d_hid)``             |                                |
  +--------------+-------------------------+                                |
  | :math:`W^z`  | ``(d_emb, d_hid)``      |                                |
  +--------------+-------------------------+                                |
  | :math:`b^z`  | ``(d_emb)``             |                                |
  +--------------+-------------------------+--------------------------------+

  - :math:`E` is the token embedding lookup table and :math:`e\t` is the token embedding of :math:`x\t`.

    - Note that the weight of :py:class:`torch.nn.Embedding` has shape ``(vocab_size, d_emb)``, which is different from
      the formula above.  The difference only affect the implementation details.

  - :math:`h\t` represent the recurrent nodes in the Elman Net model.  The initial hidden state :math:`h[0]` is a
    trainable parameter.

  - :math:`z\t` serve as the projection of :math:`h\t` from dimension ``d_hid`` to ``d_emb``.

  - The final output :math:`y\t` is the next token id prediction probability distribution.  We use inner product to
    calculate similarity scores over all token ids, and then use softmax to normalize similarity scores into
    probability range :math:`[0, 1]`.

  Parameters
  ----------
  d_emb: int
    Token embedding dimension.
  d_hid: int
    Hidden states dimension.
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.
  p_emb: float
    Embeddings dropout probability.
  p_hid: float
    Hidden units dropout probability.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance.

  Attributes
  ----------
  emb: torch.nn.Embedding
    Token embedding lookup table.
  h_0: torch.nn.Parameter
    Initial hidden states.
  model_name: ClassVar[str]
    CLI name of Elman Net is ``elman-net``.
  proj_e2h: torch.nn.Sequential
    Fully connected layer which connects input units to recurrent units.  Input dimension is ``d_emb`` and output
    dimension is ``d_hid``.
  proj_h2h: torch.nn.Linear
    Fully connected layer which connects recurrent units to recurrent units.  Input and output dimensions are ``d_hid``.
  proj_h2e: torch.nn.Sequential
    Fully connected layer which project hidden states to embedding dimension.  Input dimension is ``d_hid`` and output
    dimension is ``d_emb``.

  References
  ----------
  .. [1] Elman, J. L. (1990). `Finding Structure in Time`_. Cognitive science, 14(2), 179-211.

  .. _`Finding Structure in Time`:
     https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1
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

    # Token embedding layer.  Use token ids to perform token embeddings lookup.
    self.emb = nn.Embedding(num_embeddings=tknzr.vocab_size, embedding_dim=d_emb, padding_idx=PAD_TKID)

    # Fully connected layer which connects input units to recurrent units.
    self.proj_e2h = nn.Sequential(
      nn.Dropout(p=p_emb),
      nn.Linear(in_features=d_emb, out_features=d_hid),
    )

    # Fully connected layer which connects recurrent units to recurrent units.  Set `bias=False` to share bias term with
    # `self.proj_e2h` layer.
    self.proj_h2h = nn.Linear(in_features=d_hid, out_features=d_hid, bias=False)

    # Initial hidden states.  First dimension is set to `1` to broadcast along batch dimension.
    self.h_0 = nn.Parameter(torch.zeros(1, d_hid))

    # Fully connected layer which project hidden states to embedding dimension.
    self.proj_h2e = nn.Sequential(
      nn.Dropout(p=p_hid),
      nn.Linear(in_features=d_hid, out_features=d_emb),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

    # Initialize model parameters.
    self.params_init()

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\frac{-1}{\sqrt{v}}, \frac{1}{\sqrt{v}}}` where :math:`v =` ``d_emb``.

    Returns
    -------
    None
    """
    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(self.emb.embedding_dim)
    nn.init.uniform_(self.emb.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_e2h[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_e2h[1].bias, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_h2h.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.h_0, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_h2e[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.proj_h2e[1].bias, -inv_sqrt_dim, inv_sqrt_dim)

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
    group = parser.add_argument_group(f'`lmp.model.{cls.__name__}` constructor arguments')
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

  def forward(self, batch_cur_tkids: torch.Tensor, batch_next_tkids: torch.Tensor) -> torch.Tensor:
    """Calculate language model training loss.

    This method must only be used to train model.  For inference use :py:meth:`lmp.model.ElmanNet.pred` instead.
    Forward pass algorithm is structured as follow:

    #. Use token ids to lookup token embeddings with ``self.emb``.
    #. Dropout token embeddings with ``self.emb_drop``.
    #. Use ``self.proj_e2h`` and ``self.proj_h2h`` to calculate recurrent units.  In this step we use teacher forcing,
       i.e., inputs are directly given instead generated by model.
    #. Use ``self.proj_h2e`` to project hidden states to embedding dimension.
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

    # Token embedding lookup and project to recurrent units.
    # In  shape: (batch_size, seq_len).
    # Out shape: (batch_size, seq_len, d_hid).
    e = self.proj_e2h(self.emb(batch_cur_tkids))

    # Perform recurrent calculation for `seq_len` steps.
    h_all = []
    h_prev: Union[torch.Tensor, nn.Parameter] = self.h_0
    for i in range(seq_len):
      # `e[:, i, :]` is the current input.  We use teacher forcing.
      # shape: (batch_size, d_hid).
      # `h_prev` is the previous hidden states.
      # shape: (batch_size, d_hid).
      # `h_cur` is the current hidden states.
      # shape: (batch_size, d_hid).
      h_cur = torch.tanh(e[:, i, :] + self.proj_h2h(h_prev))

      h_all.append(h_cur)
      h_prev = h_cur

    # Stack list of tensors into single tensor.
    # In  shape: list of (batch_size, d_hid) with length equals to `seq_len`.
    # Out shape: (batch_size, seq_len, d_hid).
    h = torch.stack(h_all, dim=1)

    # Project hidden states to embedding dimension.
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

    This method must only be used for inference.  For training use :py:meth:`lmp.model.ElmanNet.forward` instead.  No
    tensor graphs will be constructed and no gradients will be calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.  ``batch_cur_tkids`` has shape ``(batch_size)`` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous calculation results.  Set to ``None`` to use initial hidden states.  ``batch_prev_states`` must
      only has one item with shape ``(batch_size, d_emb)`` and ``dtype == torch.float``.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the tensor of batch of next token id probability distribution with shape
      ``(batch_size, vocab_size)`` and ``dtype == torch.float``.  The second item in the tuple is a list of tensor
      which represent current hiddent states.  There is only one tensor in the list, and it has shape
      ``(batch_size, d_hid)`` and ``dtype == torch.float``.
    """
    # Use initial hidden state if `batch_prev_state is None`.
    if batch_prev_states is None:
      batch_prev_states = [self.h_0]

    # Token embedding lookup and project to recurrent units.
    # In  shape: (batch_size).
    # Out shape: (batch_size, d_hid).
    e = self.proj_e2h(self.emb(batch_cur_tkids))

    # Perform recurrent calculation.
    # shape: (batch_size, d_hid).
    h_prev = batch_prev_states[0]
    h_cur = torch.tanh(e + self.proj_h2h(h_prev))

    # Project hidden states to embedding dimension.
    # shape: (batch_size, d_emb)
    z = self.proj_h2e(h_cur)

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # shape: (batch_size, vocab_size).
    sim = z @ self.emb.weight.transpose(0, 1)

    # Calculate next token id probability distribution using softmax.
    # shape: (batch_size, vocab_size).
    batch_next_tkids_pd = F.softmax(sim, dim=1)

    return (batch_next_tkids_pd, [h_cur])
