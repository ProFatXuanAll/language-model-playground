"""Elman Net language model."""

import argparse
from typing import Any, ClassVar, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp.util.validate
from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr
from lmp.vars import PAD_TKID


class ElmanNet(BaseModel):
  r"""Elman Net :footcite:`elman1990finding` language model.

  - Let :math:`x` be batch of token ids with batch size :math:`B` and per sequence length :math:`S`.
  - Let :math:`V` be the vocabulary size of the paired tokenizer.
    Each token id represents an unique token, i.e., :math:`x_t \in \set{1, \dots, V}`.
  - Let :math:`E` be the token embedding lookup table.

    - Let :math:`\dEmb` be the dimension of token embeddings.
    - Let :math:`e_t` be the token embedding correspond to token id :math:`x_t`.
    - Token embeddings have dropout probability :math:`\pEmb`.

  - Let :math:`\nLyr` be the number of recurrent layers.
  - Let :math:`\dHid` be the number of recurrent units in each recurrent layer.
  - Let :math:`h^\ell` be the hidden states of the :math:`\ell` th recurrent layer.

    - Let :math:`h_t^\ell` be the :math:`t` th time step of :math:`h^\ell`.
    - The initial hidden states :math:`h_0^\ell` are given as input.
    - Hidden states have dropout probability :math:`\pHid`.

  Elman Net language model is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\ElmanNet}\pa{x, \br{h_0^1, \dots, h_0^{\nLyr}}}                                        \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                                      \\
      & \indent{2} e_t \algoEq (x_t)\text{-th row of } E \text{ but treated as column vector}             \\
      & \indent{2} \widehat{e_t} \algoEq \drop{e_t}{\pEmb}                                                \\
      & \indent{2} h_t^0 \algoEq \tanh\pa{W_h \cdot \widehat{e_t} + b_h}                                  \\
      & \indent{1} \algoEndFor                                                                            \\
      & \indent{1} h^0 \algoEq \cat{h_1^0, \dots, h_S^0}                                                  \\
      & \indent{1} \widehat{h^0} \algoEq \drop{h^0}{\pHid}                                                \\
      & \indent{1} \algoFor{\ell \in \set{1, \dots, \nLyr}}                                               \\
      & \indent{2} h^\ell \algoEq \ElmanNetLayer\pa{x \algoEq \widehat{h^{\ell-1}}, h_0 \algoEq h_0^\ell} \\
      & \indent{2} \widehat{h^\ell} \algoEq \drop{h^\ell}{\pHid}                                          \\
      & \indent{1} \algoEndFor                                                                            \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                                      \\
      & \indent{2} z_t \algoEq \tanh\pa{W_z \cdot h_t^{\nLyr} + b_z}                                      \\
      & \indent{2} \widehat{z_t} \algoEq \drop{z_t}{\pHid}                                                \\
      & \indent{2} y_t \algoEq \sof{E \cdot \widehat{z_t}}                                                \\
      & \indent{1} \algoEndFor                                                                            \\
      & \indent{1} y \algoEq \cat{y_1, \dots, y_S}                                                        \\
      & \indent{1} \algoReturn \pa{y, \br{h_S^1, \dots, h_S^{\nLyr}}}                                     \\
      & \algoEndProc
    \end{align*}

  +-------------------------------------------+--------------------------------------------------+
  | Trainable Parameters                      | Nodes                                            |
  +------------------+------------------------+--------------------------+-----------------------+
  | Parameter        | Shape                  | Symbol                   | Shape                 |
  +==================+========================+==========================+=======================+
  | :math:`E`        | :math:`(V, \dEmb)`     | :math:`e_t`              | :math:`(B, S, \dEmb)` |
  +------------------+------------------------+--------------------------+-----------------------+
  | :math:`W_h`      | :math:`(\dHid, \dEmb)` | :math:`\widehat{e_t}`    | :math:`(B, S, \dEmb)` |
  +------------------+------------------------+--------------------------+-----------------------+
  | :math:`W_z`      | :math:`(\dEmb, \dHid)` | :math:`h^\ell`           | :math:`(B, S, \dHid)` |
  +------------------+------------------------+--------------------------+-----------------------+
  | :math:`b_h`      | :math:`(\dHid)`        | :math:`h_t^\ell`         | :math:`(B, \dHid)`    |
  +------------------+------------------------+--------------------------+-----------------------+
  | :math:`b_z`      | :math:`(\dEmb)`        | :math:`\widehat{h^\ell}` | :math:`(B, S, \dHid)` |
  +------------------+------------------------+--------------------------+-----------------------+
  | :math:`\ElmanNetLayer`                    | :math:`x`                | :math:`(B, S)`        |
  +------------------+------------------------+--------------------------+-----------------------+
  |                                           | :math:`x_t`              | :math:`(B)`           |
  |                                           +--------------------------+-----------------------+
  |                                           | :math:`y`                | :math:`(B, S, V)`     |
  |                                           +--------------------------+-----------------------+
  |                                           | :math:`y_t`              | :math:`(B, V)`        |
  |                                           +--------------------------+-----------------------+
  |                                           | :math:`z_t`              | :math:`(B, \dEmb)`    |
  |                                           +--------------------------+-----------------------+
  |                                           | :math:`\widehat{z_t}`    | :math:`(B, \dEmb)`    |
  +-------------------------------------------+--------------------------+-----------------------+

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
  - Our implementation use :math:`\tanh` as activation function instead of the sigmoid function as used in the paper.
    The consideration here is simply to allow embeddings have negative values.
  - Model parameters in Elman Net are initialized with uniform distribution
    :math:`\mathcal{U}(\init_l, \init_u)`.
    The lower bound :math:`\init_l` and upper bound :math:`\init_u` of uniform distribution are given as
    hyperparameters.

  Parameters
  ----------
  d_emb: int, default: 1
    Token embedding dimension :math:`\dEmb`.
  d_hid: int, default: 1
    Hidden states dimension :math:`\dHid`.
  init_lower: float, default: -0.1
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float, default: 0.1
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  label_smoothing: float, default: 0.0
    Smoothing applied on prediction target :math:`x_{t+1}`.
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
  d_emb: int
    Token embedding dimension :math:`\dEmb`.
  d_hid: int
    Hidden states dimension :math:`\dHid`.
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
  init_lower: float, default: -0.1
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float, default: 0.1
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  label_smoothing: float, default: 0.0
    Smoothing applied on prediction target :math:`x_{t+1}`.
  loss_fn: torch.nn.CrossEntropyLoss
    Loss function to be optimized.
  model_name: ClassVar[str]
    CLI name of Elman Net is ``Elman-Net``.
  n_lyr: int
    Number of recurrent layers :math:`\nLyr`.
  p_emb: float
    Embeddings dropout probability :math:`\pEmb`.
  p_hid: float
    Hidden units dropout probability :math:`\pHid`.
  stack_rnn: torch.nn.ModuleList
    :py:class:`~ElmanNetLayer` stacking layers.
    Each Elman Net layer is followed by a dropout layer with probability :math:`\pHid`.
    The number of stacking layers is equal to :math:`2 \nLyr`.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dHid)`.

  See Also
  --------
  ~ElmanNetLayer
    Elman Net recurrent neural network.
  """

  model_name: ClassVar[str] = 'Elman-Net'

  def __init__(
    self,
    *,
    d_emb: int = 1,
    d_hid: int = 1,
    init_lower: float = -0.1,
    init_upper: float = 0.1,
    label_smoothing: float = 0.0,
    n_lyr: int = 1,
    p_emb: float = 0.0,
    p_hid: float = 0.0,
    tknzr: BaseTknzr,
    **kwargs: Any,
  ):
    super().__init__(**kwargs)

    # `d_emb` validation.
    lmp.util.validate.raise_if_not_instance(val=d_emb, val_name='d_emb', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_emb], val_names=['1', 'd_emb'])
    self.d_emb = d_emb

    # `d_hid` validation.
    lmp.util.validate.raise_if_not_instance(val=d_hid, val_name='d_hid', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_hid], val_names=['1', 'd_hid'])
    self.d_hid = d_hid

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
      nn.Linear(in_features=d_emb, out_features=d_hid),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

    # Stacking Elman Net layers.
    # Each RNN layer is followed by one dropout layer.
    self.stack_rnn = nn.ModuleList([])
    for _ in range(n_lyr):
      self.stack_rnn.append(
        ElmanNetLayer(
          in_feat=d_hid,
          init_lower=init_lower,
          init_upper=init_upper,
          out_feat=d_hid,
        )
      )
      self.stack_rnn.append(nn.Dropout(p=p_hid))

    # Fully connected layer which transforms hidden states to next token embeddings.
    # Dropout is applied to make transform robust.
    self.fc_h2e = nn.Sequential(
      nn.Linear(in_features=d_hid, out_features=d_emb),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

    # Loss function used to optimize language model.
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TKID, label_smoothing=label_smoothing)

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add Elman Net language model hyperparameters to CLI argument parser.

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
    >>> from lmp.model import ElmanNet
    >>> parser = argparse.ArgumentParser()
    >>> ElmanNet.add_CLI_args(parser)
    >>> args = parser.parse_args([
    ...   '--d_emb', '2',
    ...   '--d_hid', '4',
    ...   '--init_lower', '-0.01',
    ...   '--init_upper', '0.01',
    ...   '--label_smoothing', '0.1',
    ...   '--n_lyr', '2',
    ...   '--p_emb', '0.5',
    ...   '--p_hid', '0.1',
    ... ])
    >>> assert args.d_emb == 2
    >>> assert args.d_hid == 4
    >>> assert math.isclose(args.init_lower, -0.01)
    >>> assert math.isclose(args.init_upper, 0.01)
    >>> assert math.isclose(args.label_smoothing, 0.1)
    >>> assert args.n_lyr == 2
    >>> assert math.isclose(args.p_emb, 0.5)
    >>> assert math.isclose(args.p_hid, 0.1)
    """
    # `parser` validation.
    lmp.util.validate.raise_if_not_instance(val=parser, val_name='parser', val_type=argparse.ArgumentParser)

    # Add hyperparameters to CLI arguments.
    group = parser.add_argument_group('Elman Net language model hyperparameters')
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
      '--d_hid',
      default=1,
      help='''
      Number of recurrent units.
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
      '--n_lyr',
      default=1,
      help='''
      Number of Elman Net layers.
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
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states.
      There are :math:`\nLyr` tensors in the list.
      Each tensor in the list has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states of each layer.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the mini-batch cross-entropy loss.
      Loss tensor has shape :math:`(1)` and ``dtype == torch.float``.
      The second item in the tuple is a list of tensor.
      Each tensor in the list is the last hiddent states of each recurrent layer derived from current input token ids.
      Each tensor in the list has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
    """
    # Get next token id logits and last hidden states.
    # Logits shape: (B, S, V).
    # Each tensor in `batch_cur_states` has shape: (B, d_hid).
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate cross-entropy loss.
    # We reshape `logits` to (B x S, V) and `batch_next_tkids` to (B x S).
    # This is needed since this is how PyTorch design its API.
    # shape: (1).
    loss = self.loss_fn(logits.reshape(-1, self.emb.num_embeddings), batch_next_tkids.reshape(-1))

    # Return loss and last hidden states.
    return (loss, batch_cur_states)

  def forward(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    r"""Calculate next token id logits.

    Logits were calculated based on previous hidden states and current input token ids.
    Use :py:meth:`~pred` to convert logits into next token id probability distribution over tokenizer's vocabulary.
    Use :py:meth:`~cal_loss` to convert logits into next token id prediction loss.
    Below we describe the forward pass algorithm of Elman Net language model.

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
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states.
      There are :math:`\nLyr` tensors in the list.
      Each tensor in the list has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states of each layer.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id logits with shape :math:`(B, S, V)` and
      ``dtype == torch.float``.
      The second item in the tuple is a list of tensor.
      Each tensor in the list is the last hiddent states of each recurrent layer derived from current input token ids.
      Each tensor in the list has shape :math:`(B, \dHid)` and ``dtype == torch.float``.

    See Also
    --------
    ~lmp.tknzr.BaseTknzr.enc
      Source of token ids.
    """
    # Use initial hidden states if `batch_prev_states is None`.
    if batch_prev_states is None:
      batch_prev_states = [None] * self.n_lyr

    # Lookup token embeddings and feed to recurrent units.
    # In  shape: (B, S).
    # Out shape: (B, S, d_hid).
    rnn_in = self.fc_e2h(self.emb(batch_cur_tkids))

    # Loop through each layer and gather last hidden states of each layer.
    batch_cur_states = []
    for lyr in range(self.n_lyr):
      # Fetch previous hidden states of a layer.
      # Shape: (B, S, d_hid).
      h_0 = batch_prev_states[lyr]

      # Get the `lyr`-th RNN layer and the `lyr`-th dropout layer.
      rnn_lyr = self.stack_rnn[2 * lyr]
      dropout_lyr = self.stack_rnn[2 * lyr + 1]

      # Use previous RNN layer's output as next RNN layer's input.
      # In  shape: (B, S, d_hid).
      # Out shape: (B, S, d_hid).
      rnn_out = rnn_lyr(x=rnn_in, h_0=h_0)

      # Record the last hidden states.
      # This must be done when no dropout is applied.
      batch_cur_states.append(rnn_out[:, -1, :].detach())

      # Apply dropout to the output.
      # In  shape: (B, S, d_hid).
      # Out shape: (B, S, d_hid).
      rnn_out = dropout_lyr(rnn_out)

      # Update RNN layer's input.
      rnn_in = rnn_out

    # Transform hidden states to next token embeddings.
    # Shape: (B, S, d_emb).
    z = self.fc_h2e(rnn_out)

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # Shape: (B, S, V).
    sim = z @ self.emb.weight.transpose(0, 1)
    return (sim, batch_cur_states)

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases are initialized with uniform distribution :math:`\mathcal{U}(\init_l, \init_u)`.

    Returns
    -------
    None

    See Also
    --------
    ~ElmanNetLayer.params_init
      Elman Net layer parameter initialization.
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
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    r"""Calculate next token id probability distribution over tokenizer's vocabulary.

    Probabilities were calculated based on previous hidden states and current input token id.
    This method is only used for inference.
    No tensor graphs are constructed and no gradients are calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states.
      There are :math:`\nLyr` tensors in the list.
      Each tensor in the list has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states of each layer.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id probability distributions over the paired tokenizer's
      vocabulary.
      Probability tensor has shape :math:`(B, S, V)` and ``dtype == torch.float``.
      The second item in the tuple is a list of tensor.
      Each tensor in the list is the last hiddent states of each recurrent layer derived from current input token ids.
      Each tensor in the list has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
    """
    # Get next token id logits and last hidden states.
    # Logits shape: (B, S, V).
    # Each tensor in `batch_cur_states` has shape: (B, d_hid).
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate next token id probability distribution using softmax.
    # shape: (B, S, V).
    return (F.softmax(logits, dim=-1), batch_cur_states)


class ElmanNetLayer(nn.Module):
  r"""Elman Net :footcite:`elman1990finding` recurrent neural network.

  - Let :math:`\hIn` be the number of input features per time step.
  - Let :math:`\hOut` be the number of output features per time step.
  - Let :math:`x` be a batch of sequence of input features with shape :math:`(B, S, \hIn)`, where :math:`B` is batch
    size and :math:`S` is per sequence length.
  - Let :math:`h_0` be the initial hidden states with shape :math:`(B, \hOut)`.

  Elman Net layer is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\ElmanNetLayer}\pa{x, h_0}                               \\
      & \indent{1} S \algoEq x.\text{size}(1)                              \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                       \\
      & \indent{2} h_t \algoEq \tanh\pa{W \cdot x_t + U \cdot h_{t-1} + b} \\
      & \indent{1} \algoEndFor                                             \\
      & \indent{1} h \algoEq \cat{h_1, \dots, h_S}                         \\
      & \indent{1} \algoReturn h                                           \\
      & \algoEndProc
    \end{align*}

  +--------------------------------------+-------------------------------------+
  | Trainable Parameters                 | Nodes                               |
  +-------------+------------------------+-------------+-----------------------+
  | Parameter   | Shape                  | Symbol      | Shape                 |
  +=============+========================+=============+=======================+
  | :math:`U`   | :math:`(\hOut, \hOut)` | :math:`h`   | :math:`(B, S, \hOut)` |
  +-------------+------------------------+-------------+-----------------------+
  | :math:`W`   | :math:`(\hOut, \hIn)`  | :math:`h_t` | :math:`(B, \hOut)`    |
  +-------------+------------------------+-------------+-----------------------+
  | :math:`b`   | :math:`(\hOut)`        | :math:`x`   | :math:`(B, S, \hIn)`  |
  +-------------+------------------------+-------------+-----------------------+
  |                                      | :math:`x_t` | :math:`(B, \hIn)`     |
  +--------------------------------------+-------------+-----------------------+

  - Our implementation use :math:`\tanh` as activation function instead of the sigmoid function as used in the paper.
    The consideration here is simply to allow embeddings have negative values.
  - Model parameters in Elman Net layer are initialized with uniform distribution :math:`\mathcal{U}(\init_l, \init_u)`.
    The lower bound :math:`\init_l` and upper bound :math:`\init_u` are given as hyperparameters.

  Parameters
  ----------
  in_feat: int, default: 1
    Number of input features per time step :math:`\hIn`.
  init_lower: float, default: -0.1
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float, default: 0.1
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  out_feat: int, default: 1
    Number of output features per time step :math:`\hOut`.

  Attributes
  ----------
  fc_x2h: torch.nn.Linear
    Fully connected layer with parameters :math:`W` and :math:`b` which connects input units to recurrent units.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \hOut)`.
  fc_h2h: torch.nn.Linear
    Fully connected layer :math:`U` which connects recurrent units to recurrent units.
    Input shape: :math:`(B, \hOut)`.
    Output shape: :math:`(B, \hOut)`.
  h_0: torch.Tensor
    Initial hidden states :math:`h_0`.
    Shape: :math:`(1, \hOut)`
  in_feat: int
    Number of input features per time step :math:`\hIn`.
  init_lower: float
    Uniform distribution lower bound :math:`\init_l` used to initialize model parameters.
  init_upper: float
    Uniform distribution upper bound :math:`\init_u` used to initialize model parameters.
  out_feat: int
    Number of output features per time step :math:`\hOut`.
  """

  def __init__(
    self,
    *,
    in_feat: int = 1,
    init_lower: float = -0.1,
    init_upper: float = 0.1,
    out_feat: int = 1,
    **kwargs: Any,
  ):
    super().__init__()

    # `in_feat` validation.
    lmp.util.validate.raise_if_not_instance(val=in_feat, val_name='in_feat', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, in_feat], val_names=['1', 'in_feat'])
    self.in_feat = in_feat

    # `init_lower` and `init_upper` validation.
    lmp.util.validate.raise_if_not_instance(val=init_lower, val_name='init_lower', val_type=float)
    lmp.util.validate.raise_if_not_instance(val=init_upper, val_name='init_upper', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[init_lower, init_upper], val_names=['init_lower', 'init_upper'])
    self.init_upper = init_upper
    self.init_lower = init_lower

    # `out_feat` validation.
    lmp.util.validate.raise_if_not_instance(val=out_feat, val_name='out_feat', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, out_feat], val_names=['1', 'out_feat'])
    self.out_feat = out_feat

    # Fully connected layer which connects input units to recurrent units.
    self.fc_x2h = nn.Linear(in_features=in_feat, out_features=out_feat)

    # Fully connected layer which connects recurrent units to recurrent units.
    # Set `bias=False` to share bias term with `self.fc_x2h` layer.
    self.fc_h2h = nn.Linear(in_features=out_feat, out_features=out_feat, bias=False)

    # Initial hidden states.
    # First dimension is set to `1` to so that ``self.h_0`` can broadcast along batch dimension.
    self.register_buffer(name='h_0', tensor=torch.zeros(1, out_feat))

  def forward(
    self,
    x: torch.Tensor,
    h_0: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    r"""Calculate batch of hidden states for ``x``.

    Below we describe the forward pass algorithm of Elman Net layer.

    #. Let ``x`` be a batch of sequences of input features :math:`x`.
    #. Let ``x.size(1)`` be sequence length :math:`S`.
    #. Let ``h_0`` be the initial hidden states :math:`h_0`.
       If ``h_0 is None``, use ``self.h_0`` instead.
    #. Loop through :math:`\set{1, \dots, S}` with looping index :math:`t`.

       #. Use ``self.fc_x2h`` to transform input features at current time steps :math:`x_t`.
       #. Use ``self.fc_h2h`` to transform recurrent units at previous time steps :math:`h_{t-1}`.
       #. Add the transformations and use :math:`\tanh` as activation function to get the recurrent units at current
          time steps :math:`h_t`.

    #. Denote the concatenation of hidden states :math:`h_1, \dots, h_S` as :math:`h`.
    #. Return :math:`h`.

    Parameters
    ----------
    x: torch.Tensor
      Batch of sequences of input features.
      ``x`` has shape :math:`(B, S, \hIn)` and ``dtype == torch.float``.
    h_0: torch.Tensor, default: None
      Batch of previous hidden states.
      The tensor has shape :math:`(B, \hOut)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states ``self.h_0``.

    Returns
    -------
    torch.Tensor
      Batch of current hidden states :math:`h`.
      Returned tensor has shape :math:`(B, S, \hOut)` and ``dtype == torch.float``.
    """
    if h_0 is None:
      h_prev = self.h_0
    else:
      h_prev = h_0

    # Sequence length.
    S = x.size(1)

    # Transform input features.
    # Shape: (B, S, out_feat).
    x2h = self.fc_x2h(x)

    # Perform recurrent calculation for `S` steps.
    h_all = []
    for t in range(S):
      # `x2h[:, t, :]` is the batch input features at time step `t`.
      # Shape: (B, out_feat).
      # `h_prev` is the hidden states at time step `t`.
      # Shape: (B, out_feat).
      # `h_cur` is the hidden states at time step `t + 1`.
      # Shape: (B, out_feat).
      h_cur = torch.tanh(x2h[:, t, :] + self.fc_h2h(h_prev))

      h_all.append(h_cur)
      h_prev = h_cur

    # Stack list of tensors into single tensor.
    # In  shape: list of (B, out_feat) with length equals to `S`.
    # Out shape: (B, S, out_feat).
    h = torch.stack(h_all, dim=1)
    return h

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases are initialized with uniform distribution :math:`\mathcal{U}(\init_l, \init_u)`.

    Returns
    -------
    None
    """
    # Initialize weights and biases with uniform distribution.
    nn.init.uniform_(self.fc_x2h.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_x2h.bias, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_h2h.weight, self.init_lower, self.init_upper)
