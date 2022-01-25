"""Elman Net language model."""

import argparse
from typing import Any, ClassVar, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp.util.validate
from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr


class ElmanNet(BaseModel):
  r"""Elman Net [1]_ language model.

  Implement RNN model in the paper `Finding Structure in Time`_.

  - Let :math:`x[t]` be the :math:`t`-th token id in the input token id list :math:`x` as defined in
    :py:class:`lmp.model.BaseModel`.
  - Let ``d_emb`` be the dimension of token embeddings and let ``vocab_size`` be the vocabulary size of tokenizer.

  Then Elman Net is defined as follow:

  .. math::

     \newcommand{\pa}[1]{\left( #1 \right)}
     \newcommand{\set}[1]{\left\lbrace #1 \right\rbrace}
     \newcommand{\sigmoid}[1]{\operatorname{sigmoid}\pa{#1}}
     \newcommand{\softmax}[1]{\operatorname{softmax}\pa{#1}}
     \begin{align*}
       e[t]     & = (x[t])\text{-th column of } E             \\
       h[t + 1] & = \sigmoid{W \cdot e[t] + U \cdot h[t] + b} \\
       y[t + 1] & = \softmax{E^{\top} \cdot h[t + 1]}
     \end{align*}

  +----------------------------------------+----------------------------------------+
  | Trainable Parameters                   | Nodes                                  |
  +--------------+-------------------------+------------------+---------------------+
  | Parameter    | Shape                   | Symbol           | Shape               |
  +==============+=========================+==================+=====================+
  | :math:`E`    | ``(d_emb, vocab_size)`` | :math:`e[t]`     | ``(d_emb, 1)``      |
  +--------------+-------------------------+------------------+---------------------+
  | :math:`h[0]` | ``(d_emb, 1)``          |                                        |
  +--------------+-------------------------+------------------+---------------------+
  | :math:`W`    | ``(d_emb, d_emb)``      | :math:`h[t + 1]` | ``(d_emb, 1)``      |
  +--------------+-------------------------+------------------+---------------------+
  | :math:`U`    | ``(d_emb, d_emb)``      | :math:`y[t + 1]` | ``(vocab_size, 1)`` |
  +--------------+-------------------------+------------------+---------------------+
  | :math:`b`    | ``(d_emb, 1)``          |                                        |
  +--------------+-------------------------+------------------+---------------------+

  - :math:`E` is the token embedding lookup table and :math:`e[t]` is the token embedding of :math:`x[t]`.

    - Note that the weight of :py:class:`torch.nn.Embedding` has shape ``(vocab_size, d_emb)``, which is different from
      the formula above.  The difference only affect the implementation details.

  - :math:`h[t + 1]` is the hidden state at time step :math:`t + 1`.  The initial hidden state :math:`h[0]` is a
    pre-defined column vector.

  - The final output :math:`y[t + 1]` is the next token id prediction probability distribution.  We use inner product
    to calculate similarity scores over all token ids, and then use softmax to normalize similarity scores into
    probability range :math:`[0, 1]`.

  Parameters
  ----------
  d_emb: int
    Token embedding dimension.
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance.

  Attributes
  ----------
  emb: torch.nn.Embedding
    Token embedding lookup table.
  h_0: torch.nn.Parameter
    Initial hidden states.
  loss_fn: torch.nn.CrossEntropyLoss
    Loss function to be optimized.
  model_name: ClassVar[str]
    CLI name of Elman Net is ``elman-net``.
  proj_e2h: torch.nn.Linear
    Fully connected layer which connects input units to hidden units.  Input and output dimensions are ``d_emb``.
  proj_h2h: torch.nn.Linear
    Fully connected layer which connects hidden units to hidden units.  Input and output dimensions are ``d_emb``.

  References
  ----------
  .. [1] Elman, J. L. (1990). `Finding Structure in Time`_. Cognitive science, 14(2), 179-211.

  .. _`Finding Structure in Time`:
     https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1
  """

  model_name: ClassVar[str] = 'Elman-Net'

  def __init__(self, *, d_emb: int, tknzr: BaseTknzr, **kwargs: Any):
    super().__init__(**kwargs)
    # `d_emb` validation.
    lmp.util.validate.raise_if_not_instance(val=d_emb, val_name='d_emb', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_emb], val_names=['1', 'd_emb'])

    # `tknzr` validation.
    lmp.util.validate.raise_if_not_instance(val=tknzr, val_name='tknzr', val_type=BaseTknzr)

    # Token embedding layer.  Use token ids to perform token embeddings lookup.
    self.emb = nn.Embedding(num_embeddings=tknzr.vocab_size, embedding_dim=d_emb, padding_idx=tknzr.pad_tkid)

    # Fully connected layer which connects input units to hidden units.
    self.proj_e2h = nn.Linear(in_features=d_emb, out_features=d_emb)

    # Fully connected layer which connects hidden units to hidden units.  Set `bias=False` to share bias term with
    # `self.proj_e2h` layer.
    self.proj_h2h = nn.Linear(in_features=d_emb, out_features=d_emb, bias=False)

    # Initial hidden states.  First dimension is set to `1` to broadcast along batch dimension.
    self.h_0 = nn.Parameter(torch.zeros(1, d_emb))

    # Calculate cross entropy loss for all non-padding tokens.
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=tknzr.pad_tkid)

  def forward(self, batch_cur_tkids: torch.Tensor, batch_next_tkids: torch.Tensor) -> torch.Tensor:
    """Calculate language model training loss.

    This method must only be used to train model.  For inference use :py:meth:`lmp.model.ElmanNet.pred` instead.
    Forward pass algorithm is structured as follow:

    #. Use token ids to lookup token embeddings with ``self.emb``.
    #. Use ``self.proj_e2h`` and ``self.proj_h2h`` to calculate recurrent units.  In this step we use teacher forcing,
       i.e., inputs are directly given instead generated by model.
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

    # Token embedding lookup.
    # In  shape: (batch_size, seq_len).
    # Out shape: (batch_size, seq_len, d_emb).
    e = self.emb(batch_cur_tkids)

    # Project from embedding layer to hidden layer.
    # shape: (batch_size, seq_len, d_emb).
    e = self.proj_e2h(e)

    # Perform recurrent calculation for `seq_len` steps.
    h_all = []
    h_prev: Union[torch.Tensor, nn.Parameter] = self.h_0
    for i in range(seq_len):
      # `e[:, i, :]` is the current input.  We use teacher forcing.
      # shape: (batch_size, d_emb).
      # `h_prev` is the previous hidden states.
      # shape: (batch_size, d_emb).
      # `h_cur` is the current hidden states.
      # shape: (batch_size, d_emb).
      h_cur = torch.sigmoid(e[:, i, :] + self.proj_h2h(h_prev))

      h_all.append(h_cur)
      h_prev = h_cur

    # Stack list of tensors into single tensor.
    # In  shape: list of (batch_size, d_emb) with length equals to `seq_len`.
    # Out shape: (batch_size, seq_len, d_emb).
    h = torch.stack(h_all, dim=1)

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # shape: (batch_size, seq_len, vocab_size).
    sim = h @ self.emb.weight.transpose(0, 1)

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
      Batch of previous calculation results.  Set to ``None`` to use initial hidden states.  ``batch_prev_states`` must
      only has one item with shape ``(batch_size, d_emb)`` and ``dtype == torch.float``.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the tensor of batch of next token id probability distribution with shape
      ``(batch_size, vocab_size)`` and ``dtype == torch.float``.  The second item in the tuple is a list of tensor
      which represent current hiddent states.  There is only one tensor in the list, and it has shape
      ``(batch_size, d_emb)`` and ``dtype == torch.float``.
    """
    # Use initial hidden state if `batch_prev_state is None`.
    if batch_prev_states is None:
      batch_prev_states = [self.h_0]

    # Token embedding lookup.
    # In  shape: (batch_size).
    # Out shape: (batch_size, d_emb).
    e = self.emb(batch_cur_tkids)

    # Project from embedding layer to hidden layer.
    # shape: (batch_size, d_emb).
    e = self.proj_e2h(e)

    # Perform recurrent calculation.
    # shape: (batch_size, d_emb).
    h_prev = batch_prev_states[0]
    h_cur = torch.sigmoid(e + self.proj_h2h(h_prev))

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # shape: (batch_size, vocab_size).
    sim = h_cur @ self.emb.weight.transpose(0, 1)

    # Calculate next token id probability distribution using softmax.
    # shape: (batch_size, vocab_size).
    batch_next_tkids_pd = F.softmax(sim, dim=-1)

    return (batch_next_tkids_pd, [h_cur])

  @classmethod
  def train_parser(cls, parser: argparse.ArgumentParser) -> None:
    """CLI arguments parser for training Elman Net language model.

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
    >>> from lmp.model import ElmanNet
    >>> parser = argparse.ArgumentParser()
    >>> ElmanNet.train_parser(parser)
    >>> args = parser.parse_args([
    ...   '--batch_size', '32',
    ...   '--beta1', '0.9',
    ...   '--beta2', '0.99',
    ...   '--ckpt_step', '1000',
    ...   '--d_emb', '100',
    ...   '--dset_name', 'wiki-text-2',
    ...   '--eps', '1e-8',
    ...   '--exp_name', 'my_exp',
    ...   '--log_step', '200',
    ...   '--lr', '1e-4',
    ...   '--max_norm', '1',
    ...   '--max_seq_len', '128',
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
    group = parser.add_argument_group('Elman Net training arguments')
    group.add_argument(
      '--d_emb',
      help='Token embedding dimension.',
      required=True,
      type=int,
    )
