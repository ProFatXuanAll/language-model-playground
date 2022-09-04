"""LSTM (1997 version) language model."""

import argparse
from typing import Any, ClassVar, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp.util.metric
import lmp.util.validate
from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr
from lmp.vars import PAD_TKID


class LSTM1997(BaseModel):
  r"""LSTM (1997 version) :footcite:`hochreiter1997lstm` language model.

  - Let :math:`x` be batch of token ids with batch size :math:`B` and per sequence length :math:`S`.
  - Let :math:`V` be the vocabulary size of the paired tokenizer.
    Each token id represents an unique token, i.e., :math:`x_t \in \set{1, \dots, V}`.
  - Let :math:`E` be the token embedding lookup table.

    - Let :math:`\dEmb` be the dimension of token embeddings.
    - Let :math:`e_t` be the token embedding correspond to token id :math:`x_t`.
    - Token embeddings have dropout probability :math:`\pEmb`.

  - Let :math:`\nLyr` be the number of recurrent layers.
  - Let :math:`\dBlk` be the number of units in a memory cell block.
  - Let :math:`\nBlk` be the number of memory cell blocks.
  - Let :math:`\dHid = \nBlk \times \dBlk`.
  - Let :math:`h^\ell` be the hidden states of the :math:`\ell` th recurrent layer.

    - Let :math:`h_t^\ell` be the :math:`t` th time step of :math:`h^\ell`.
    - The initial hidden states :math:`h_0^\ell` are given as input.
    - Hidden states have dropout probability :math:`\pHid`.

  - Let :math:`c^\ell` be the memory cell internal states of the :math:`\ell` th recurrent layer.

    - let :math:`c_t^\ell` be the :math:`t` th time step of :math:`c^\ell`.
    - The memory cell initial internal states :math:`c_0^\ell` are given as input.

  LSTM (1997 version) language model is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\LSTMNineSeven}\pa{x, \pa{\br{c_0^1, \dots, c_0^{\nLyr}}, \br{h_0^1, \dots, h_0^{\nLyr}}}} \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                                         \\
      & \indent{2} e_t \algoEq (x_t)\text{-th row of } E \text{ but treated as column vector}                \\
      & \indent{2} \widehat{e_t} \algoEq \drop{e_t}{\pEmb}                                                   \\
      & \indent{2} h_t^0 \algoEq \tanh\pa{W_h \cdot \widehat{e_t} + b_h}                                     \\
      & \indent{1} \algoEndFor                                                                               \\
      & \indent{1} h^0 \algoEq \cat{h_1^0, \dots, h_S^0}                                                     \\
      & \indent{1} \widehat{h^0} \algoEq \drop{h^0}{\pHid}                                                   \\
      & \indent{1} \algoFor{\ell \in \set{1, \dots, \nLyr}}                                                  \\
      & \indent{2} \pa{c^\ell, h^\ell} \algoEq \LSTMNineSevenLayer\pa{
                                                  x \algoEq \widehat{h^{\ell-1}},
                                                  c_0 \algoEq c_0^\ell,
                                                  h_0 \algoEq h_0^\ell
                                               }                                                             \\
      & \indent{2} \widehat{h^\ell} \algoEq \drop{h^\ell}{\pHid}                                             \\
      & \indent{1} \algoEndFor                                                                               \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                                         \\
      & \indent{2} z_t \algoEq \tanh\pa{W_z \cdot h_t^{\nLyr} + b_z}                                         \\
      & \indent{2} \widehat{z_t} \algoEq \drop{z_t}{\pHid}                                                   \\
      & \indent{2} y_t \algoEq \sof{E \cdot \widehat{z_t}}                                                   \\
      & \indent{1} \algoEndFor                                                                               \\
      & \indent{1} y \algoEq \cat{y_1, \dots, y_S}                                                           \\
      & \indent{1} \algoReturn \pa{y, \pa{\br{c_S^1, \dots, c_S^{\nLyr}}, \br{h_S^1, \dots, h_S^{\nLyr}}}}   \\
      & \algoEndProc
    \end{align*}

  +-------------------------------------------+---------------------------------------------------------+
  | Trainable Parameters                      | Nodes                                                   |
  +------------------+------------------------+--------------------------+------------------------------+
  | Parameter        | Shape                  | Symbol                   | Shape                        |
  +==================+========================+==========================+==============================+
  | :math:`E`        | :math:`(V, \dEmb)`     | :math:`c^\ell`           | :math:`(B, S, \nBlk, \dBlk)` |
  +------------------+------------------------+--------------------------+------------------------------+
  | :math:`W_h`      | :math:`(\dHid, \dEmb)` | :math:`c_t^\ell`         | :math:`(B, \nBlk, \dBlk)`    |
  +------------------+------------------------+--------------------------+------------------------------+
  | :math:`W_z`      | :math:`(\dEmb, \dHid)` | :math:`e_t`              | :math:`(B, S, \dEmb)`        |
  +------------------+------------------------+--------------------------+------------------------------+
  | :math:`b_h`      | :math:`(\dHid)`        | :math:`h^\ell`           | :math:`(B, S, \dHid)`        |
  +------------------+------------------------+--------------------------+------------------------------+
  | :math:`b_z`      | :math:`(\dEmb)`        | :math:`h_t^\ell`         | :math:`(B, \dHid)`           |
  +------------------+------------------------+--------------------------+------------------------------+
  | :math:`\LSTMNineSevenLayer`               | :math:`\widehat{h^\ell}` | :math:`(B, \dHid)`           |
  +------------------+------------------------+--------------------------+------------------------------+
  |                                           | :math:`x`                | :math:`(B, S)`               |
  |                                           +--------------------------+------------------------------+
  |                                           | :math:`x_t`              | :math:`(B)`                  |
  |                                           +--------------------------+------------------------------+
  |                                           | :math:`y`                | :math:`(B, S, V)`            |
  |                                           +--------------------------+------------------------------+
  |                                           | :math:`y_t`              | :math:`(B, V)`               |
  |                                           +--------------------------+------------------------------+
  |                                           | :math:`z_t`              | :math:`(B, \dEmb)`           |
  |                                           +--------------------------+------------------------------+
  |                                           | :math:`\widehat{z_t}`    | :math:`(B, \dEmb)`           |
  +-------------------------------------------+--------------------------+------------------------------+

  The goal of optimization is to minimize the negative logliklihood of next token id :math:`x_{t+1}` given :math:`y_t`.
  The prediction loss is defined to be the average negative logliklihood over :math:`x` given :math:`y`.

  .. math::

    \loss = \dfrac{-1}{S} \sum_{t = 1}^S \log \Pr(x_{t+1} \vert y_t).

  - :math:`z_t` is obtained by transforming :math:`h_t^{\nLyr}` from dimension :math:`\dHid` to :math:`\dEmb`.
    This is only need for shape consistency:
    the hidden states :math:`h_t^{\nLyr}` has shape :math:`(B, \dHid)` and :math:`E` has shape :math:`(V, \dEmb)`.
  - :math:`y_t` is the next token id prediction probability distribution over tokenizer's vocabulary.
    We use inner product to calculate similarity scores over all token ids, and then use softmax to normalize
    similarity scores into probability range :math:`[0, 1]`.
  - The calculations after hidden states are the same as :py:class:`~lmp.model.ElmanNet`.
  - Model parameters in LSTM (1997 version) are initialized with uniform distribution
    :math:`\mathcal{U}(\init_l, \init_u)`.
    The lower bound :math:`\init_l` and upper bound :math:`\init_u` of uniform distribution are given as
    hyperparameters.
  - Input gate biases are initialized with uniform distribution :math:`\mathcal{U}(\init_{ib}, 0)`.
    This make input gate remain closed at the start of training.
  - Output gate biases are initialized with uniform distribution :math:`\mathcal{U}(\init_{ob}, 0)`.
    This make output gate remain closed at the start of training.

  Parameters
  ----------
  d_blk: int, default: 1
    Number of units in a memory cell block :math:`\dBlk`.
  d_emb: int, default: 1
    Token embedding dimension :math:`\dEmb`.
  init_ib: float, default: -1.0
    Uniform distribution lower bound :math:`\init_{ib}` used to initialize input gate biases.
  init_lower: float, default: -0.1
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_ob: float, default: -1.0
    Uniform distribution lower bound :math:`\init_{ob}` used to initialize output gate biases.
  init_upper: float, default: 0.1
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  label_smoothing: float, default: 0.0
    Smoothing applied on prediction target :math:`x_{t+1}`.
  n_blk: int, default: 1
    Number of memory cell blocks :math:`\nBlk`.
  n_lyr: int, default: 1
    Number of recurrent layers :math:`\nLyr`.
  p_emb: float, default: 0.0
    Embeddings dropout probability :math:`\pEmb`.
  p_hid: float, default: 0.0
    Hidden units dropout probability :math:`\pHid`.
  tknzr: lmp.tknzr.BaseTknzr
    Tokenizer instance.

  Attributes
  ----------
  d_blk: int
    Number of units in a memory cell block :math:`\dBlk`.
  d_hid: int
    Total number of memory cell units :math:`\dHid`.
  emb: torch.nn.Embedding
    Token embedding lookup table :math:`E`.
    Input shape: :math:`(B, S)`.
    Output shape: :math:`(B, S, \dEmb)`.
  fc_e2h: torch.nn.Sequential
    Fully connected layer :math:`W_h` and :math:`b_h` which connects input
    units to the 1st recurrent layer's input.
    Dropout with probability :math:`\pEmb` is applied to input.
    Dropout with probability :math:`\pHid` is applied to output.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \dHid)`.
  fc_h2e: torch.nn.Sequential
    Fully connected layer :math:`W_z` and :math:`b_z` which transforms hidden states to next token embeddings.
    Dropout with probability :math:`\pHid` is applied to output.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dEmb)`.
  init_ib: float
    Uniform distribution lower bound :math:`\init_{ib}` used to initialize input gate biases.
  init_lower: float
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_ob: float
    Uniform distribution lower bound :math:`\init_{ob}` used to initialize output gate biases.
  init_upper: float
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  label_smoothing: float
    Smoothing applied on prediction target :math:`x_{t+1}`.
  model_name: ClassVar[str]
    CLI name of LSTM (1997 version) is ``LSTM-1997``.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.
  n_lyr: int
    Number of recurrent layers :math:`\nLyr`.
  p_emb: float
    Embeddings dropout probability :math:`\pEmb`.
  p_hid: float
    Hidden units dropout probability :math:`\pHid`.
  stack_rnn: torch.nn.ModuleList
    :py:class:`~LSTM1997Layer` stacking layers.
    Each LSTM (1997 version) layer is followed by a dropout layer with probability :math:`\pHid`.
    The number of stacking layers is equal to :math:`2 \nLyr`.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dHid)`.

  See Also
  --------
  ~lmp.model.ElmanNet
    Elman Net language model.
  ~LSTM1997Layer
    LSTM (1997 version) recurrent neural network.
  """

  model_name: ClassVar[str] = 'LSTM-1997'

  def __init__(
    self,
    *,
    d_blk: int = 1,
    d_emb: int = 1,
    init_ib: float = -1.0,
    init_lower: float = -0.1,
    init_ob: float = -1.0,
    init_upper: float = 0.1,
    label_smoothing: float = 0.0,
    n_blk: int = 1,
    n_lyr: int = 1,
    p_emb: float = 0.0,
    p_hid: float = 0.0,
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
    self.d_emb = d_emb

    # `init_ib` validation.
    lmp.util.validate.raise_if_not_instance(val=init_ib, val_name='init_ib', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[init_ib, 0], val_names=['init_ib', '0'])
    self.init_ib = init_ib

    # `init_ob` validation.
    lmp.util.validate.raise_if_not_instance(val=init_ob, val_name='init_ob', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[init_ob, 0], val_names=['init_ob', '0'])
    self.init_ob = init_ob

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

    # `n_blk` validation.
    lmp.util.validate.raise_if_not_instance(val=n_blk, val_name='n_blk', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_blk], val_names=['1', 'n_blk'])
    self.n_blk = n_blk
    self.d_hid = n_blk * d_blk

    # `n_lyr` validation.
    lmp.util.validate.raise_if_not_instance(val=n_lyr, val_name='n_lyr', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_lyr], val_names=['1', 'n_lyr'])
    self.n_lyr = n_lyr

    # `p_emb` validation.
    lmp.util.validate.raise_if_not_instance(val=p_emb, val_name='p_emb', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, p_emb, 1.0], val_names=['0.0', 'p_emb', '1.0'])
    self.p_emb = p_emb

    # `p_hid` validation.
    lmp.util.validate.raise_if_not_instance(val=p_hid, val_name='p_hid', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0.0, p_hid, 1.0], val_names=['0.0', 'p_hid', '1.0'])
    self.p_hid = p_hid

    # `tknzr` validation.
    lmp.util.validate.raise_if_not_instance(val=tknzr, val_name='tknzr', val_type=BaseTknzr)

    # Token embedding layer.
    # Use token ids to perform token embeddings lookup.
    self.emb = nn.Embedding(num_embeddings=tknzr.vocab_size, embedding_dim=d_emb, padding_idx=PAD_TKID)

    # Fully connected layer which connects input units to the 1st recurrent layer's input.
    # Dropout is applied to make embeddings robust.
    self.fc_e2h = nn.Sequential(
      nn.Dropout(p=p_emb),
      nn.Linear(in_features=d_emb, out_features=self.d_hid),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

    # Stacking LSTM (1997 version) layers.
    # Each RNN layer is followed by one dropout layer.
    self.stack_rnn = nn.ModuleList([])
    for _ in range(n_lyr):
      self.stack_rnn.append(
        LSTM1997Layer(
          d_blk=d_blk,
          in_feat=self.d_hid,
          init_ib=init_ib,
          init_lower=init_lower,
          init_ob=init_ob,
          init_upper=init_upper,
          n_blk=n_blk,
        )
      )
      self.stack_rnn.append(nn.Dropout(p=p_hid))

    # Fully connected layer which transforms hidden states to next token embeddings.
    # Dropout is applied to make transform robust.
    self.fc_h2e = nn.Sequential(
      nn.Linear(in_features=self.d_hid, out_features=d_emb),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

    # Loss function used to optimize language model.
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TKID, label_smoothing=label_smoothing)

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add LSTM (1997 version) language model hyperparameters to CLI argument parser.

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
    >>> from lmp.model import LSTM1997
    >>> parser = argparse.ArgumentParser()
    >>> LSTM1997.add_CLI_args(parser)
    >>> args = parser.parse_args([
    ...   '--d_blk', '64',
    ...   '--d_emb', '100',
    ...   '--init_ib', '-0.1',
    ...   '--init_lower', '-0.01',
    ...   '--init_ob', '-0.1',
    ...   '--init_upper', '0.01',
    ...   '--label_smoothing', '0.1',
    ...   '--n_blk', '8',
    ...   '--n_lyr', '2',
    ...   '--p_emb', '0.5',
    ...   '--p_hid', '0.1',
    ... ])
    >>> assert args.d_blk == 64
    >>> assert args.d_emb == 100
    >>> assert math.isclose(args.init_ib, -0.1)
    >>> assert math.isclose(args.init_lower, -0.01)
    >>> assert math.isclose(args.init_ob, -0.1)
    >>> assert math.isclose(args.init_upper, 0.01)
    >>> assert math.isclose(args.label_smoothing, 0.1)
    >>> assert args.n_blk == 8
    >>> assert args.n_lyr == 2
    >>> assert math.isclose(args.p_emb, 0.5)
    >>> assert math.isclose(args.p_hid, 0.1)
    """
    # `parser` validation.
    lmp.util.validate.raise_if_not_instance(val=parser, val_name='parser', val_type=argparse.ArgumentParser)

    # Add hyperparameters to CLI arguments.
    group = parser.add_argument_group('LSTM (1997 version) language model hyperparameters')
    group.add_argument(
      '--d_blk',
      default=1,
      help='''
      Dimension of each memory cell block.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--d_emb',
      default=1,
      help='''
      Token embedding dimension.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--init_ib',
      default=-1.0,
      help='''
      Uniform distribution lower bound used to initialize input gate biases.
      Default is ``-1.0``.
      ''',
      type=float,
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
      '--init_ob',
      default=-1.0,
      help='''
      Uniform distribution lower bound used to initialize output gate biases.
      Default is ``-1.0``.
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
      '--n_blk',
      default=1,
      help='''
      Number of memory cell blocks.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--n_lyr',
      default=1,
      help='''
      Number of LSTM (1997 version) layers.
      Default is ``1``.
      ''',
      type=int,
    )
    group.add_argument(
      '--p_emb',
      default=0.0,
      help='''
      Embeddings dropout probability.
      Default is ``0.0``.
      ''',
      type=float,
    )
    group.add_argument(
      '--p_hid',
      default=0.0,
      help='''
      Hidden units dropout probability.
      Default is ``0.0``.
      ''',
      type=float,
    )

  def cal_loss(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
    batch_prev_states: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
  ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
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
    batch_prev_states: typing.Optional[tuple[list[torch.Tensor], list[torch.Tensor]]], default: None
      The first tensor list in the tuple is a batch of memory cell previous internal states.
      There are :math:`\nLyr` tensors in the list, each has shape :math:`(B, \nBlk, \dBlk)` and
      ``dtype == torch.float``.
      The second tensor list in the tuple is a batch of previous hidden states.
      There are :math:`\nLyr` tensors in the list, each has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states and memory cell initial internal states of each layer.

    Returns
    -------
    tuple[torch.Tensor, tuple[list[torch.Tensor], list[torch.Tensor]]]
      The first item in the tuple is the mini-batch cross-entropy loss.
      Loss tensor has shape :math:`(1)` and ``dtype == torch.float``.
      The second item in the tuple is a tuple containing two tensor lists.
      The first tensor list represents the memory cell last internal states of each recurrent layer derived from
      current input token ids.
      Each tensor in the first list has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
      The second tensor list represents the last hidden states of each recurrent layer derived from current input token
      ids.
      Each tensor in the second list has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Both structure are the same as ``batch_prev_states``.
    """
    # Get next token id logits, memory cell last internal states and last hidden states.
    # Logits shape: (B, S, V)
    # Last hidden states shapes: [(B, d_hid)]
    # Memory cell last internal states shapes: [(B, n_blk, d_blk)]
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate cross-entropy loss.
    # We reshape `logits` to (B x S, V) and `batch_next_tkids` to (B x S).
    # This is needed since this is how PyTorch design its API.
    # shape: (1).
    loss = self.loss_fn(logits.reshape(-1, self.emb.num_embeddings), batch_next_tkids.reshape(-1))

    # Return loss, memory cell last internal states and last hidden states.
    return (loss, batch_cur_states)

  def forward(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
  ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    r"""Calculate next token id logits.

    Logits were calculated based on previous hidden states, memory cell previous internal states and and current input
    token ids.
    Use :py:meth:`~pred` to convert logits into next token id probability distribution over tokenizer's vocabulary.
    Use :py:meth:`~cal_loss` to convert logits into next token id prediction loss.
    Below we describe the forward pass algorithm of LSTM (1997 version) language model.

    #. Use token ids to lookup token embeddings with ``self.emb``.
    #. Use ``self.fc_e2h`` to transform token embeddings into 1st recurrent layer's input.
    #. Feed transformation result into recurrent layer and output hidden states.
       We use teacher forcing in this step when perform training, i.e., inputs are directly given instead of generated
       by model.
    #. Feed the output of previous recurrent layer into next recurrent layer until all layers have been used once.
    #. Use ``self.fc_h2e`` to transform last recurrent layer's hidden states to next token embeddings.
    #. Perform inner product on token embeddings over tokenizer's vocabulary to get similarity scores.
    #. Return similarity scores (logits).

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[tuple[list[torch.Tensor], list[torch.Tensor]]], default: None
      The first tensor list in the tuple is a batch of memory cell previous internal states.
      There are :math:`\nLyr` tensors in the list, each has shape :math:`(B, \nBlk, \dBlk)` and
      ``dtype == torch.float``.
      The second tensor list in the tuple is a batch of previous hidden states.
      There are :math:`\nLyr` tensors in the list, each has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states and memory cell initial internal states of each layer.

    Returns
    -------
    tuple[torch.Tensor, tuple[list[torch.Tensor], list[torch.Tensor]]]
      The first item in the tuple is the batch of next token id logits with shape :math:`(B, S, V)` and
      ``dtype == torch.float``.
      The second item in the tuple is a tuple containing two tensor lists.
      The first tensor list represents the memory cell last internal states of each recurrent layer derived from
      current input token ids.
      Each tensor in the first list has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
      The second tensor list represents the last hidden states of each recurrent layer derived from current input token
      ids.
      Each tensor in the second list has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Both structure are the same as ``batch_prev_states``.
    """
    # Use initial hidden states and memory cell initial internal states if `batch_prev_states is None`.
    if batch_prev_states is None:
      batch_prev_states = [[None] * self.n_lyr, [None] * self.n_lyr]

    # Lookup token embeddings and feed to recurrent units.
    # In  shape: (B, S).
    # Out shape: (B, S, d_hid).
    rnn_in = self.fc_e2h(self.emb(batch_cur_tkids))

    # Loop through each layer and gather last hidden states and memory cell last internal states of each layer.
    batch_cur_states = ([], [])
    for lyr in range(self.n_lyr):
      # Fetch memory cell previous internal states of a layer.
      # Shape: (B, S, n_blk, d_blk).
      c_0 = batch_prev_states[0][lyr]

      # Fetch previous hidden states of a layer.
      # Shape: (B, S, d_hid).
      h_0 = batch_prev_states[1][lyr]

      # Get the `lyr`-th RNN layer and the `lyr`-th dropout layer.
      rnn_lyr = self.stack_rnn[2 * lyr]
      dropout_lyr = self.stack_rnn[2 * lyr + 1]

      # Use previous RNN layer's output as next RNN layer's input.
      # In  shape: (B, S, d_hid).
      # rnn_c_out shape: (B, S, n_blk, d_blk).
      # rnn_h_out shape: (B, S, d_hid).
      rnn_c_out, rnn_h_out = rnn_lyr(x=rnn_in, c_0=c_0, h_0=h_0)

      # Record the last internal states.
      batch_cur_states[0].append(rnn_c_out[:, -1, :, :].detach())

      # Record the last hidden states.
      batch_cur_states[1].append(rnn_h_out[:, -1, :].detach())

      # Apply dropout to the output.
      # In  shape: (B, S, d_hid).
      # Out shape: (B, S, d_hid).
      rnn_h_out = dropout_lyr(rnn_h_out)

      # Update RNN layer's input.
      rnn_in = rnn_h_out

    # Transform hidden states to next token embeddings.
    # Shape: (B, S, d_emb).
    z = self.fc_h2e(rnn_h_out)

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # Shape: (B, S, V).
    sim = z @ self.emb.weight.transpose(0, 1)
    return (sim, batch_cur_states)

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases other than input / output gate biases are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\init_l, \init_u}`.
    Input gate biases are initialized with uniform distribution :math:`\mathcal{U}\pa{\init_{ib}, 0}`.
    Output gate biases are initialized with uniform distribution :math:`\mathcal{U}\pa{\init_{ob}, 0}`.

    Returns
    -------
    None

    See Also
    --------
    ~LSTM1997Layer.params_init
      LSTM (1997 version) layer parameter initialization.
    """
    # Initialize weights and biases with uniform distribution.
    nn.init.uniform_(self.emb.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_e2h[1].weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_e2h[1].bias, self.init_lower, self.init_upper)
    for lyr in range(self.n_lyr):
      self.stack_rnn[2 * lyr].params_init()
    nn.init.uniform_(self.fc_h2e[0].weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_h2e[0].bias, self.init_lower, self.init_upper)

  @torch.no_grad()
  def pred(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
  ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    r"""Calculate next token id probability distribution over tokenizer's vocabulary.

    Probabilities were calculated based on previous hidden states, memory cell previous internal states and current
    input token id.
    This method is only used for inference.
    No tensor graphs are constructed and no gradients are calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[tuple[list[torch.Tensor], list[torch.Tensor]]], default: None
      The first tensor list in the tuple is a batch of memory cell previous internal states.
      There are :math:`\nLyr` tensors in the list, each has shape :math:`(B, \nBlk, \dBlk)` and
      ``dtype == torch.float``.
      The second tensor list in the tuple is a batch of previous hidden states.
      There are :math:`\nLyr` tensors in the list, each has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states and memory cell initial internal states of each layer.

    Returns
    -------
    tuple[torch.Tensor, tuple[list[torch.Tensor], list[torch.Tensor]]]
      The first item in the tuple is the batch of next token id probability distributions over the tokenizer's
      vocabulary.
      Probability tensor has shape :math:`(B, S, V)` and ``dtype == torch.float``.
      The second item in the tuple is a tuple containing two tensor lists.
      The first tensor list represents the memory cell last internal states of each recurrent layer derived from
      current input token ids.
      Each tensor in the first list has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
      The second tensor list represents the last hidden states of each recurrent layer derived from current input token
      ids.
      Each tensor in the second list has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Both structure are the same as ``batch_prev_states``.
    """
    # Get next token id logits, memory cell last internal states and last hidden states.
    # Logits shape: (B, S, V)
    # Last hidden states shapes: [(B, d_hid)]
    # Memory cell last internal states shapes: [(B, n_blk, d_blk)]
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate next token id probability distribution using softmax.
    # shape: (B, S, V).
    return (F.softmax(logits, dim=-1), batch_cur_states)


class LSTM1997Layer(nn.Module):
  r"""LSTM (1997 version) :footcite:`hochreiter1997lstm` recurrent neural network.

  - Let :math:`\hIn` be the number of input features per time step.
  - Let :math:`\dBlk` be the number of units in a memory cell block.
  - Let :math:`\nBlk` be the number of memory cell blocks.
  - Let :math:`\hOut = \nBlk \times \dBlk` be the number of output features per time step.
  - Let :math:`x` be a batch of sequences of input features with shape :math:`(B, S, \hIn)`, where :math:`B` is batch
    size and :math:`S` is per sequence length.
  - Let :math:`h_0` be the initial hidden states with shape :math:`(B, \hOut)`.
  - Let :math:`c_0` be the memory cell initial internal states with shape :math:`(B, \nBlk, \dBlk)`.

  LSTM (1997 version) layer is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\LSTMNineSevenLayer}\pa{x, c_0, h_0}                                              \\
      & \indent{1} S \algoEq x.\text{size}(1)                                                       \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                                \\
      & \indent{2} i_t \algoEq \sigma\pa{W_i \cdot x_t + U_i \cdot h_{t-1} + b_i}                   \\
      & \indent{2} o_t \algoEq \sigma\pa{W_o \cdot x_t + U_o \cdot h_{t-1} + b_o}                   \\
      & \indent{2} \algoFor{k \in \set{1, \dots, \nBlk}}                                            \\
      & \indent{3} g_{t,k} \algoEq \tanh\pa{W_k \cdot x_t + U_k \cdot h_t + b_k} &&\tag{1}\label{1} \\
      & \indent{3} c_{t,k} \algoEq c_{t-1,k} + i_{t,k} \cdot g_{t,k}                                \\
      & \indent{3} h_{t,k} \algoEq o_{t,k} \cdot \tanh\pa{c_{t,k}}               &&\tag{2}\label{2} \\
      & \indent{2} \algoEndFor                                                                      \\
      & \indent{2} c_t \algoEq \cat{c_{t,1}, \dots, c_{t,\nBlk}}                                    \\
      & \indent{2} h_t \algoEq \fla{h_{t,1}, \dots, h_{t,\nBlk}}                                    \\
      & \indent{1} \algoEndFor                                                                      \\
      & \indent{1} c \algoEq \cat{c_1, \dots, c_S}                                                  \\
      & \indent{1} h \algoEq \cat{h_1, \dots, h_S}                                                  \\
      & \indent{1} \algoReturn (c, h)                                                               \\
      & \algoEndProc
    \end{align*}

  +--------------------------------------+------------------------------------------------+
  | Trainable Parameters                 | Nodes                                          |
  +-------------+------------------------+-----------------+------------------------------+
  | Parameter   | Shape                  | Symbol          | Shape                        |
  +=============+========================+=================+==============================+
  | :math:`U_i` | :math:`(\nBlk, \hOut)` | :math:`c`       | :math:`(B, S, \nBlk, \dBlk)` |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`U_k` | :math:`(\dBlk, \hOut)` | :math:`c_t`     | :math:`(B, \nBlk, \dBlk)`    |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`U_o` | :math:`(\nBlk, \hOut)` | :math:`c_{t,k}` | :math:`(B, \dBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`W_i` | :math:`(\nBlk, \hIn)`  | :math:`g_{t,k}` | :math:`(B, \dBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`W_k` | :math:`(\dBlk, \hIn)`  | :math:`h`       | :math:`(B, S, \hOut)`        |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`W_o` | :math:`(\nBlk, \hIn)`  | :math:`h_t`     | :math:`(B, \hOut)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`b_i` | :math:`(\nBlk)`        | :math:`h_{t,k}` | :math:`(B, \dBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`b_o` | :math:`(\nBlk)`        | :math:`i_t`     | :math:`(B, \nBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`b_k` | :math:`(\dBlk)`        | :math:`i_{t,k}` | :math:`(B, 1)`               |
  +-------------+------------------------+-----------------+------------------------------+
  |                                      | :math:`o_t`     | :math:`(B, \nBlk)`           |
  |                                      +-----------------+------------------------------+
  |                                      | :math:`o_{t,k}` | :math:`(B, 1)`               |
  |                                      +-----------------+------------------------------+
  |                                      | :math:`x`       | :math:`(B, S, \hIn)`         |
  |                                      +-----------------+------------------------------+
  |                                      | :math:`x_t`     | :math:`(B, \hIn)`            |
  +--------------------------------------+-----------------+------------------------------+

  - :math:`i_t` is memory cell blocks' input gate units at time step :math:`t`.
    :math:`i_{t,k}` is the :math:`k`-th coordinates of :math:`i_t`, which represents the :math:`k`-th memory cell
    block's input gate unit at time step :math:`t`.
  - :math:`o_t` is memory cell blocks' output gate units at time step :math:`t`.
    :math:`o_{t,k}` is the :math:`k`-th coordinates of :math:`o_t`, which represents the :math:`k`-th memory cell
    block's output gate unit at time step :math:`t`.
  - The :math:`k`-th memory cell block is consist of:

    - Current input features :math:`x_t`.
    - Previous hidden states :math:`h_{t-1}`.
    - Input activations :math:`g_{t,k}`.
    - A input gate unit :math:`i_{t,k}`.
    - A output gate unit :math:`o_{t,k}`.
    - Memory cell previous internal states :math:`c_{t-1,k}` and memory cell current internal states :math:`c_{t,k}`.
    - Output units :math:`h_{t,k}`.

  - All memory cell current internal states at time step :math:`t` are concatenated to form :math:`c_t`.
  - All memory cell output units at time step :math:`t` are flattened to form :math:`h_t`.
  - Our implementation use :math:`\tanh` as memory cell blocks' input activation function.
    The implementation in the paper use :math:`4 \sigma - 2` in :math:`\eqref{1}` and :math:`2 \sigma - 1` in
    :math:`\eqref{2}`.
    We argue that the change in :math:`\eqref{1}` does not greatly affect the computation result and :math:`\eqref{2}`
    is almost the same as the paper implementation.
    To be precise, :math:`\tanh(x) = 2 \sigma(2x) - 1`.
    The formula :math:`2 \sigma(x) - 1` has gradient :math:`2 \sigma(x) (1 - \sigma(x))`.
    The formula :math:`\tanh(x)` has gradient :math:`4 \sigma(2x) (1 - \sigma(2x))`.
    Intuitively using :math:`\tanh` will scale gradient by 4.
  - Model parameters in LSTM (1997 version) layer are initialized with uniform distribution
    :math:`\mathcal{U}(\init_l, \init_u)`.
    The lower bound :math:`\init_l` and upper bound :math:`\init_u` are given as hyperparameters.
  - Input gate biases are initialized with uniform distribution :math:`\mathcal{U}(\init_{ib}, 0)`.
    The lower bound :math:`\init_{ib}` is given as hyperparameter.
    This make input gate remain closed at the start of training.
  - Output gate biases are initialized with uniform distribution :math:`\mathcal{U}(\init_{ob}, 0)`.
    The lower bound :math:`\init_{ob}` is given as hyperparameter.
    This make output gate remain closed at the start of training.

  Parameters
  ----------
  d_blk: int, default: 1
    Dimension of each memory cell block :math:`\dBlk`.
  in_feat: int, default: 1
    Number of input features per time step :math:`\hIn`.
  init_ib: float, default: -1.0
    Uniform distribution lower bound :math:`\init_{ib}` used to initialize input gate biases.
  init_lower: float, default: -0.1
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_ob: float, default: -1.0
    Uniform distribution lower bound :math:`\init_{ob}` used to initialize output gate biases.
  init_upper: float, default: 0.1
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  n_blk: int, default: 1
    Number of memory cell blocks :math:`\nBlk`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  c_0: torch.Tensor
    Memory cell blocks' initial internal states :math:`c_0`.
    Shape: :math:`(1, \nBlk, \dBlk)`.
  d_blk: int
    Number of units in a memory cell block :math:`\dBlk`.
  d_hid: int
    Total number of memory cell units :math:`\hOut`.
  fc_h2ig: torch.nn.Linear
    Fully connected layer :math:`U_i` which connects hidden states to memory cell's input gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  fc_h2mc_in: torch.nn.Linear
    Fully connected layers :math:`\pa{U_1, \dots, U_{\nBlk}}` which connect hidden states to memory cell
    blocks' input activations.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \dHid)`.
  fc_h2og: torch.nn.Linear
    Fully connected layer :math:`U_o` which connects hidden states to memory cell's output gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  fc_x2ig: torch.nn.Linear
    Fully connected layer :math:`W_i` and :math:`b_i` which connects input units to memory cell's input gate units.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \nBlk)`.
  fc_x2mc_in: torch.nn.Linear
    Fully connected layers :math:`\pa{W_1, \dots, W_{\nBlk}}` and :math:`\pa{b_1, \dots, b_{\nBlk}}` which connects
    input units to memory cell blocks' input activations.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \dHid)`.
  fc_x2og: torch.nn.Linear
    Fully connected layer :math:`W_o` and :math:`b_o` which connects input units to memory cell's output gate units.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \nBlk)`.
  h_0: torch.Tensor
    Initial hidden states :math:`h_0`.
    Shape: :math:`(1, \dHid)`
  in_feat: int
    Number of input features per time step :math:`\hIn`.
  init_ib: float
    Uniform distribution lower bound :math:`\init_{ib}` used to initialize input gate biases.
  init_lower: float
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_ob: float
    Uniform distribution lower bound :math:`\init_{ob}` used to initialize output gate biases.
  init_upper: float
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.
  """

  def __init__(
    self,
    *,
    d_blk: int = 1,
    in_feat: int = 1,
    init_ib: float = -1.0,
    init_lower: float = -0.1,
    init_ob: float = -1.0,
    init_upper: float = 0.1,
    n_blk: int = 1,
    **kwargs: Any,
  ):
    super().__init__()

    # `d_blk` validation.
    lmp.util.validate.raise_if_not_instance(val=d_blk, val_name='d_blk', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_blk], val_names=['1', 'd_blk'])
    self.d_blk = d_blk

    # `in_feat` validation.
    lmp.util.validate.raise_if_not_instance(val=in_feat, val_name='in_feat', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, in_feat], val_names=['1', 'in_feat'])
    self.in_feat = in_feat

    # `init_ib` validation.
    lmp.util.validate.raise_if_not_instance(val=init_ib, val_name='init_ib', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[init_ib, 0], val_names=['init_ib', '0'])
    self.init_ib = init_ib

    # `init_ob` validation.
    lmp.util.validate.raise_if_not_instance(val=init_ob, val_name='init_ob', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[init_ob, 0], val_names=['init_ob', '0'])
    self.init_ob = init_ob

    # `init_lower` and `init_upper` validation.
    lmp.util.validate.raise_if_not_instance(val=init_lower, val_name='init_lower', val_type=float)
    lmp.util.validate.raise_if_not_instance(val=init_upper, val_name='init_upper', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[init_lower, init_upper], val_names=['init_lower', 'init_upper'])
    self.init_upper = init_upper
    self.init_lower = init_lower

    # `n_blk` validation.
    lmp.util.validate.raise_if_not_instance(val=n_blk, val_name='n_blk', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_blk], val_names=['1', 'n_blk'])
    self.n_blk = n_blk
    self.d_hid = n_blk * d_blk

    # Fully connected layer which connects input units to input / output gate units.
    self.fc_x2ig = nn.Linear(in_features=in_feat, out_features=n_blk)
    self.fc_x2og = nn.Linear(in_features=in_feat, out_features=n_blk)

    # Fully connected layer which connects hidden states to input / output gate units.
    # Set `bias=False` to share bias term with `self.fc_x2ig` and `self.fc_x2og` layers.
    self.fc_h2ig = nn.Linear(in_features=self.d_hid, out_features=n_blk, bias=False)
    self.fc_h2og = nn.Linear(in_features=self.d_hid, out_features=n_blk, bias=False)

    # Fully connected layer which connects input units to memory cell blocks' input activation.
    self.fc_x2mc_in = nn.Linear(in_features=in_feat, out_features=self.d_hid)

    # Fully connected layer which connects hidden states to memory cell blocks' input activation.
    # Set `bias=False` to share bias term with `self.fc_x2mc_in` layer.
    self.fc_h2mc_in = nn.Linear(in_features=self.d_hid, out_features=self.d_hid, bias=False)

    # Initial hidden states and memory cell initial internal states.
    # First dimension is set to `1` to so that they can broadcast along batch dimension.
    self.register_buffer(name='h_0', tensor=torch.zeros(1, self.d_hid))
    self.register_buffer(name='c_0', tensor=torch.zeros(1, n_blk, d_blk))

  def forward(
    self,
    x: torch.Tensor,
    c_0: Optional[torch.Tensor] = None,
    h_0: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculate batch of hidden states for ``x``.

    Below we describe the forward pass algorithm of LSTM (1997 version) layer.

    #. Let ``x`` be a batch of sequences of input features :math:`x`.
    #. Let ``x.size(1)`` be sequence length :math:`S`.
    #. Let ``c_0`` be the memory cell initial internal states :math:`c_0`.
       If ``c_0 is None``, use ``self.c_0`` instead.
    #. Let ``h_0`` be the initial hidden states :math:`h_0`.
       If ``h_0 is None``, use ``self.h_0`` instead.
    #. Loop through :math:`\set{1, \dots, S}` with looping index :math:`t`.

       #. Use :math:`x_t`, :math:`h_{t-1}`, ``self.fc_x2ig`` and ``self.fc_h2ig`` to get input gate units :math:`i_t`.
       #. Use :math:`x_t`, :math:`h_{t-1}`, ``self.fc_x2og`` and ``self.fc_h2og`` to get output gate units :math:`o_t`.
       #. Use :math:`x_t`, :math:`h_{t-1}`, ``self.fc_x2mc_in`` and ``self.fc_h2mc_in`` to get memory cell input
          activations :math:`g_{t,1}, \dots, g_{t,\nBlk}`.
       #. Derive memory cell new internal state :math:`c_{t,1}, \dots, c_{t,\nBlk}` using input gate units
          :math:`i_{t,1}, \dots, i_{t,\nBlk}`, memory cell old internal states :math:`c_{t-1,1}, \dots, c_{t-1,\nBlk}`
          and memory cell input activations :math:`g_{t,1}, \dots, g_{t,\nBlk}`.
       #. Derive new hidden states :math:`h_t` using output gate units :math:`o_{t,1}, \dots, o_{t,\nBlk}` and
          memory cell new internal states :math:`c_{t,1}, \dots, c_{t,\nBlk}`.

    #. Denote the concatenation of memory cell internal states :math:`c_1, \dots, c_S` as :math:`c`.
    #. Denote the concatenation of hidden states :math:`h_1, \dots, h_S` as :math:`h`.
    #. Return :math:`(c, h)`.

    Parameters
    ----------
    x: torch.Tensor
      Batch of sequences of input features.
      ``x`` has shape :math:`(B, S, \hIn)` and ``dtype == torch.float``.
    c_0: typing.Optional[torch.Tensor], default: None
      Batch of memory cell previous internal states.
      The tensor has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial memory internal state ``self.c_0``.
    h_0: typing.Optional[torch.Tensor], default: None
      Batch of previous hidden states.
      The tensor has shape :math:`(B, \hOut)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states ``self.h_0``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
      The first tensor is batch of memory cell internal states and the second tensor is batch of hidden states.
      Batch memory cell internal states has shape :math:`(B, S, \nBlk, \dBlk)` and ``dtype == torch.float``.
      Batch hidden states has shape :math:`(B, S, \hOut)` and ``dtype == torch.float``.
    """
    if c_0 is None:
      c_prev = self.c_0
    else:
      c_prev = c_0

    if h_0 is None:
      h_prev = self.h_0
    else:
      h_prev = h_0

    # Sequence length.
    S = x.size(1)

    # Transform input features to gate units.
    # Shape: (B, S, n_blk).
    x2ig = self.fc_x2ig(x)
    x2og = self.fc_x2og(x)

    # Transform input features to memory cell block's input.
    # Shape: (B, S, d_hid).
    x2mc_in = self.fc_x2mc_in(x)

    # Perform recurrent calculation for `S` steps.
    c_all = []
    h_all = []
    for t in range(S):
      # Get input / output gate units and unsqueeze to separate memory cell blocks.
      # Shape: (B, n_blk, 1).
      ig = torch.sigmoid(x2ig[:, t, :] + self.fc_h2ig(h_prev)).unsqueeze(-1)
      og = torch.sigmoid(x2og[:, t, :] + self.fc_h2og(h_prev)).unsqueeze(-1)

      # Calculate memory cell blocks input activation and reshape to separate memory cell blocks.
      # Shape: (B, n_blk, d_blk).
      mc_in = torch.tanh(x2mc_in[:, t, :] + self.fc_h2mc_in(h_prev)).reshape(-1, self.n_blk, self.d_blk)

      # Calculate memory cell new internal states.
      # Shape: (B, n_blk, d_blk).
      c_cur = c_prev + ig * mc_in

      # Calculate memory cell outputs and flatten to form the new hidden states.
      # Shape: (B, d_hid).
      h_cur = (og * torch.tanh(c_cur)).reshape(-1, self.d_hid)

      c_all.append(c_cur)
      h_all.append(h_cur)

      # Update hidden states and memory cell internal states.
      c_prev = c_cur
      h_prev = h_cur

    # Stack list of tensors into single tensor.
    # In  shape: list of (B, n_blk, d_blk) with length equals to `S`.
    # Out shape: (B, S, n_blk, d_blk).
    c = torch.stack(c_all, dim=1)

    # Stack list of tensors into single tensor.
    # In  shape: list of (B, d_hid) with length equals to `S`.
    # Out shape: (B, S, d_hid).
    h = torch.stack(h_all, dim=1)

    return (c, h)

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases other than :math:`b_i, b_o` are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\init_l, \init_u}`.
    :math:`b_i` is initialized with uniform distribution :math:`\mathcal{U}\pa{\init_{ib}, 0}`.
    :math:`b_o` is initialized with uniform distribution :math:`\mathcal{U}\pa{\init_{ob}, 0}`.
    :math:`b_i, b_o` are initialized separatedly so that input and output gates remain closed at the start of training.

    Returns
    -------
    None
    """
    # Initialize weights and biases with uniform distribution.
    nn.init.uniform_(self.fc_x2ig.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_x2og.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_h2ig.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_h2og.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_x2mc_in.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_x2mc_in.bias, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_h2mc_in.weight, self.init_lower, self.init_upper)

    # Gate units' biases are initialized to negative values.
    nn.init.uniform_(self.fc_x2ig.bias, self.init_ib, 0.0)
    nn.init.uniform_(self.fc_x2og.bias, self.init_ob, 0.0)
