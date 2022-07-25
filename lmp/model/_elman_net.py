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


class ElmanNetLayer(nn.Module):
  r"""Elman Net [1]_ recurrent neural network.

  Implement RNN model in the paper `Finding Structure in Time`_.

  .. _`Finding Structure in Time`: https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1

  Let :math:`x` be input features with shape :math:`(B, S, H)`, where :math:`B` is batch size, :math:`S` is sequence
  length and :math:`H` is the number of features per time step in each sequence.
  Let :math:`h_0` be the initial hidden states.

  Elman Net layer is defined as follow:

  .. math::

    \newcommand{\pa}[1]{\left( #1 \right)}
    \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
    \newcommand{\cat}[1]{\operatorname{concate}\pa{#1}}
    \begin{align*}
      & \textbf{procedure } \text{ElmanNetLayer}(x, h_0)                  \\
      & \quad  S \leftarrow x.\text{size}(1)                              \\
      & \quad  \textbf{for } t \in \set{0, \dots, S-1} \textbf{ do}       \\
      & \qquad h_{t+1} \leftarrow \tanh\pa{W \cdot x_t + U \cdot h_t + b} \\
      & \quad  \textbf{end for}                                           \\
      & \quad  h \leftarrow \cat{h_1, \dots, h_{t+1}}                     \\
      & \quad  \textbf{return } h                                         \\
      & \textbf{end procedure}
    \end{align*}

  +------------------------------+---------------------------------+
  | Trainable Parameters         | Nodes                           |
  +-------------+----------------+-------------+-------------------+
  | Parameter   | Shape          | Symbol      | Shape             |
  +=============+================+=============+===================+
  | :math:`h_0` | :math:`(1, H)` | :math:`h_0` | :math:`(B, H)`    |
  +-------------+----------------+-------------+-------------------+
  | :math:`W`   | :math:`(H, H)` | :math:`x`   | :math:`(B, S, H)` |
  +-------------+----------------+-------------+-------------------+
  | :math:`U`   | :math:`(H, H)` | :math:`x_t` | :math:`(B, H)`    |
  +-------------+----------------+-------------+-------------------+
  | :math:`b`   | :math:`(H)`    | :math:`h_t` | :math:`(B, H)`    |
  +-------------+----------------+-------------+-------------------+
  |                              | :math:`h`   | :math:`(B, S, H)` |
  +------------------------------+-------------+-------------------+

  - Our implementation use :math:`\tanh` as activation function instead of the sigmoid function as used in the paper.
    The consideration here is simply to allow embeddings have negative values.

  Parameters
  ----------
  n_feat: int
    Number of input features :math:`H`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  fc_x2h: torch.nn.Sequential
    Fully connected layer with parameters :math:`W` and :math:`b` which connects input units to recurrent units.
    Input shape: :math:`(B, S, H)`.
    Output shape: :math:`(B, S, H)`.
  fc_h2h: torch.nn.Linear
    Fully connected layer :math:`U` which connects recurrent units to recurrent units.
    Input shape: :math:`(B, H)`.
    Output shape: :math:`(B, H)`.
  h_0: torch.nn.Parameter
    Initial hidden states :math:`h_0`.
    Shape: :math:`(1, H)`
  n_feat: int
    Number of input features :math:`H`.

  References
  ----------
  .. [1] Elman, J. L. (1990). `Finding Structure in Time`_. Cognitive science, 14(2), 179-211.
  """

  def __init__(self, n_feat: int, **kwargs: Any):
    super().__init__()

    # `n_feat` validation.
    lmp.util.validate.raise_if_not_instance(val=n_feat, val_name='n_feat', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_feat], val_names=['1', 'n_feat'])
    self.n_feat = n_feat

    # Fully connected layer which connects input units to recurrent units.
    self.fc_x2h = nn.Linear(in_features=n_feat, out_features=n_feat)

    # Fully connected layer which connects recurrent units to recurrent units.
    # Set `bias=False` to share bias term with `self.fc_x2h` layer.
    self.fc_h2h = nn.Linear(in_features=n_feat, out_features=n_feat, bias=False)

    # Initial hidden states.
    # First dimension is set to `1` to so that ``self.h_0`` can broadcast along batch dimension.
    self.h_0 = nn.Parameter(torch.zeros(1, n_feat))

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{H}}, \dfrac{1}{\sqrt{H}}}`.

    Returns
    -------
    None
    """
    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(self.n_feat)

    nn.init.uniform_(self.fc_x2h.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_x2h.bias, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2h.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.h_0, -inv_sqrt_dim, inv_sqrt_dim)

  def forward(
    self,
    batch_x: torch.Tensor,
    batch_prev_states: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    r"""Calculate batch of hidden states for ``batch_x``.

    Below we describe the forward pass algorithm of Elman Net layer.

    #. Let ``batch_x`` be batch input features :math:`x`.
    #. Let ``batch_prev_states`` be the initial hidden states :math:`h_0`.
       If ``batch_prev_states is None``, use ``self.h_0`` instead.
    #. Let ``batch_x.size(1)`` be sequence length :math:`S`.
    #. Loop through :math:`\set{0, \dots, S-1}` with looping index :math:`t`.

       #. Use ``self.fc_x2h`` and ``self.fc_h2h`` to transform current time steps' input features :math:`x_t` and
          previous time steps' recurrent units :math:`h_t`, respectively.
       #. Add the transformations and use :math:`\tanh` as activation function to get the current time steps' recurrent
          units :math:`h_{t+1}`.

    #. Denote the concatenation of hidden states :math:`h_1, \dots, h_S` as :math:`h`.
    #. Return :math:`h`.

    Parameters
    ----------
    batch_x: torch.Tensor
      Batch of input features.
      ``batch_x`` has shape :math:`(B, S, H)` and ``dtype == torch.float``.
    batch_prev_states: typing.Optional[torch.Tensor], default: None
      Batch of previous hidden states.
      ``batch_prev_states`` has shape :math:`(B, H)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states :math:`h_0`.

    Returns
    -------
    torch.Tensor
      batch of hidden states with shape :math:`(B, S, H)` and ``dtype == torch.float``.
    """
    if batch_prev_states is None:
      batch_prev_states = self.h_0

    h_prev = batch_prev_states

    # Sequence length.
    S = batch_x.size(1)

    # Transform input features.
    # shape: (B, S, H).
    batch_x2h = self.fc_x2h(batch_x)

    # Perform recurrent calculation for `S` steps.
    h_all = []
    for t in range(S):
      # `batch_x[:, t, :]` is the batch of input features at time step `t`.
      # shape: (B, H).
      # `h_prev` is the hidden states at time step `t`.
      # shape: (B, H).
      # `h_cur` is the hidden states at time step `t + 1`.
      # shape: (B, H).
      h_cur = torch.tanh(batch_x2h[:, t, :] + self.fc_h2h(h_prev))

      h_all.append(h_cur)
      h_prev = h_cur

    # Stack list of tensors into single tensor.
    # In  shape: list of (B, H) with length equals to `S`.
    # Out shape: (B, S, H).
    h = torch.stack(h_all, dim=1)
    return h


class ElmanNet(BaseModel):
  r"""Elman Net language model.

  Implement RNN model in the paper `Finding Structure in Time`_ as a language model.

  .. _`Finding Structure in Time`: https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1

  - Let :math:`x` be batch of token ids with shape :math:`(B, S)`, where :math:`B` is batch size and :math:`S` is
    sequence length.
  - Let :math:`V` be the vocabulary size of the paired tokenizer.
    Each token id represents an unique token, i.e., :math:`x_t \in \set{0, \dots, V -1}`.
  - Let :math:`E` be the token embedding lookup table.

    - Let :math:`\newcommand{\dEmb}{d_{\operatorname{emb}}} \dEmb` be the dimension of token embeddings.
    - Let :math:`e_t` be the token embedding correspond to token id :math:`x_t`.

  - Let :math:`\newcommand{\nLyr}{n_{\operatorname{lyr}}} \nLyr` be the number of recurrent layers and let
    :math:`\newcommand{\dHid}{d_{\operatorname{hid}}} \dHid` be the number of recurrent units in each recurrent layer.
  - Let :math:`h^\ell` be the hidden states of the :math:`\ell` th recurrent layer, let :math:`h_t^\ell` be the
    :math:`t` th time step of :math:`h^\ell` and let :math:`h_0^\ell` be the initial hidden states of the :math:`\ell`
    th recurrent layer.
    The initial hidden states are given as input.

  Elman Net language model is defined as follow:

  .. math::

    \newcommand{\pa}[1]{\left( #1 \right)}
    \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
    \newcommand{\cat}[1]{\operatorname{concate}\pa{#1}}
    \begin{align*}
      & \textbf{procedure }\text{ElmanNet}(x, [h_0^0, \dots, h_0^{\nLyr-1}])                  \\
      & \quad  \textbf{for } t \in \set{0, \dots, S-1} \textbf{ do}                           \\
      & \qquad  e_t \leftarrow (x_t)\text{-th row of } E \text{ but treated as column vector} \\
      & \qquad  h_t^{-1} \leftarrow \tanh\pa{W_h \cdot e_t + b_h}                             \\
      & \quad  \textbf{end for}                                                               \\
      & \quad  h^{-1} \leftarrow \cat{h_0^{-1}, \dots, h_{S-1}^{-1}}                          \\
      & \quad  \textbf{for } \ell \in \set{0, \dots, \nLyr-1} \textbf{ do}                    \\
      & \qquad  h^\ell \leftarrow \text{ElmanNetLayer}(x=h^{\ell-1}, h_0=h_0^\ell)            \\
      & \quad  \textbf{end for}                                                               \\
      & \quad  \textbf{for } t \in \set{0, \dots, S-1} \textbf{ do}                           \\
      & \qquad  z_{t+1} \leftarrow \tanh\pa{W_z \cdot h_{t+1}^{\nLyr-1} + b_z}                \\
      & \qquad  y_{t+1} \leftarrow \sof{E \cdot z_{t+1}}                                      \\
      & \quad  \textbf{end for}                                                               \\
      & \quad  y \leftarrow \cat{y_1, \dots, y_S}                                             \\
      & \quad  \textbf{return } y                                                             \\
      & \textbf{end procedure}
    \end{align*}

  +-------------------------------------------+--------------------------------------------+
  | Trainable Parameters                      | Nodes                                      |
  +------------------+------------------------+------------------+-------------------------+
  | Parameter        | Shape                  | Symbol           | Shape                   |
  +==================+========================+==================+=========================+
  | :math:`h_0^\ell` | :math:`(1, \dHid)`     | :math:`h_0^\ell` | :math:`(B, \dHid)`      |
  +------------------+------------------------+------------------+-------------------------+
  | :math:`E`        | :math:`(V, \dEmb)`     | :math:`x_t`      | :math:`(B, S)`          |
  +------------------+------------------------+------------------+-------------------------+
  | :math:`W_h`      | :math:`(\dHid, \dEmb)` | :math:`e_t`      | :math:`(B, S, \dEmb)`   |
  +------------------+------------------------+------------------+-------------------------+
  | :math:`b_h`      | :math:`(\dHid)`        | :math:`h_t^{-1}` | :math:`(B, \dHid)`      |
  +------------------+------------------------+------------------+-------------------------+
  | :math:`W_z`      | :math:`(\dEmb, \dHid)` | :math:`h^{-1}`   | :math:`(B, S, \dHid)`   |
  +------------------+------------------------+------------------+-------------------------+
  | :math:`b_z`      | :math:`(\dEmb)`        | :math:`h^\ell`   | :math:`(B, S, \dHid)`   |
  +------------------+------------------------+------------------+-------------------------+
  | :math:`\text{ElmanNetLayer}`              | :math:`h_t^\ell` | :math:`(B, \dHid)`      |
  +-------------------------------------------+------------------+-------------------------+
  |                                           | :math:`z_t`      | :math:`(B, \dEmb)`      |
  |                                           +------------------+-------------------------+
  |                                           | :math:`y_t`      | :math:`(B, V)`          |
  |                                           +------------------+-------------------------+
  |                                           | :math:`y`        | :math:`(B, S, V)`       |
  +-------------------------------------------+------------------+-------------------------+

  - :math:`z_{t+1}` is obtained by transforming :math:`h_{t+1}^{\nLyr-1}` from dimension :math:`\dHid` to :math:`\dEmb`.
    This is only need for shape consistency:
    the hidden states :math:`h_{t+1}^{\nLyr-1}` has shape :math:`(B, \dHid)`, and the final output :math:`y_{t+1}` has
    shape :math:`(B, V)`.
  - :math:`y_{t+1}` is the next token id prediction probability distribution over tokenizer's vocabulary.
    We use inner product to calculate similarity scores over all token ids, and then use softmax to normalize
    similarity scores into probability range :math:`[0, 1]`.
  - Our implementation use :math:`\tanh` as activation function instead of the sigmoid function as used in the paper.
    The consideration here is simply to allow embeddings have negative values.

  Parameters
  ----------
  d_emb: int
    Token embedding dimension :math:`\dEmb`.
  d_hid: int
    Hidden states dimension :math:`\dHid`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.
  n_lyr: int
    Number of recurrent layers :math:`\nLyr`.
  p_emb: float
    Embeddings dropout probability.
  p_hid: float
    Hidden units dropout probability.
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
    units to the 0th recurrent layer's input.
    Dropout with probability ``p_emb`` is applied to input.
    Dropout with probability ``p_hid`` is applied to output.
    Input shape: :math:`(B, S, \dEmb)`.
    Output shape: :math:`(B, S, \dHid)`.
  fc_h2e: torch.nn.Sequential
    Fully connected layer :math:`W_z` and :math:`b_z` which transforms hidden states to next token embeddings.
    Dropout with probability ``p_hid`` is applied to output.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dEmb)`.
  model_name: ClassVar[str]
    CLI name of Elman Net is ``Elman-Net``.
  n_lyr: int
    Number of recurrent layers :math:`\nLyr`.
  p_emb: float
    Embeddings dropout probability.
  p_hid: float
    Hidden units dropout probability.
  stack_rnn: torch.nn.ModuleList
    :py:class:`lmp.model.ElmanNetLayer` stacking layers.
    Each Elman Net layer is followed by a dropout layer with probability ``p_hid``.
    The number of stacking layers is equal to ``2 * n_lyr``.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dHid)`.

  See Also
  --------
  lmp.model.ElmanNetLayer
    Elman Net recurrent neural network.
  """

  model_name: ClassVar[str] = 'Elman-Net'

  def __init__(
    self,
    *,
    d_emb: int,
    d_hid: int,
    n_lyr: int,
    p_emb: float,
    p_hid: float,
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

    # Fully connected layer which connects input units to the 0th recurrent layer's input.
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
      self.stack_rnn.append(ElmanNetLayer(n_feat=d_hid))
      self.stack_rnn.append(nn.Dropout(p=p_hid))

    # Fully connected layer which transforms hidden states to next token embeddings.
    # Dropout is applied to make transform robust.
    self.fc_h2e = nn.Sequential(
      nn.Linear(in_features=d_hid, out_features=d_emb),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases other than :py:class:`lmp.model.ElmanNetLayer` are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, \dfrac{1}{\sqrt{d}}}` where :math:`d = \max(\dEmb, \dHid)`.
    :py:class:`lmp.model.ElmanNetLayer` are initialized by :py:meth:`lmp.model.ElmanNetLayer.params_init`.

    Returns
    -------
    None

    See Also
    --------
    lmp.model.ElmanNetLayer.params_init
      Elman Net layer parameter initialization.
    """
    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(max(self.d_emb, self.d_hid))

    nn.init.uniform_(self.emb.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2h[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2h[1].bias, -inv_sqrt_dim, inv_sqrt_dim)
    for lyr in range(self.n_lyr):
      self.stack_rnn[2 * lyr].params_init()
    nn.init.uniform_(self.fc_h2e[0].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2e[0].bias, -inv_sqrt_dim, inv_sqrt_dim)

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
    ...   '--n_lyr', '2',
    ...   '--p_emb', '0.5',
    ...   '--p_hid', '0.1',
    ... ])
    >>> assert args.d_emb == 2
    >>> assert args.d_hid == 4
    >>> assert args.n_lyr == 2
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
      '--n_lyr',
      help='Number of Elman Net layers.',
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
    #. Use ``self.fc_e2h`` to transform token embeddings into 0th recurrent layer's input.
    #. Feed transformation result into recurrent layer and output hidden states.
       In this step we use teacher forcing, i.e., inputs are directly given instead of generated by model.
    #. Feed the output of previous recurrent layer into next recurrent layer until all layers have been used once.
    #. Use ``self.fc_h2e`` to transform last recurrent layer's hidden states to next token embeddings.
    #. Perform inner product on token embeddings over tokenizer's vocabulary to get similarity scores.
    #. Return similarity scores (logits).

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states.
      There are ``n_lyr`` tensors in the list, the shape of each tensor is :math:`(B, \dHid)` and
      ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states of each layer.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id logits with shape :math:`(B, S, V)` and
      ``dtype == torch.float``.
      The second item in the tuple is a list of tensor.
      Each tensor in the list is the last hiddent states of each recurrent layer derived from current input token ids.
      Each tensor has shape :math:`(B, \dHid)` and ``dtype == torch.float``.

    See Also
    --------
    lmp.tknzr.BaseTknzr.enc
      Source of token ids.
    """
    # Use initial hidden states if `batch_prev_states is None`.
    if batch_prev_states is None:
      batch_prev_states = [None] * self.n_lyr

    # Lookup token embeddings and feed to recurrent units.
    # In  shape: (B, S).
    # Out shape: (B, S, d_hid).
    rnn_lyr_in = self.fc_e2h(self.emb(batch_cur_tkids))

    # Loop through each layer and gather the last hidden states of each layer.
    batch_cur_states = []
    for lyr in range(self.n_lyr):
      # Fetch previous hidden state of a layer.
      # Shape: (B, S).
      rnn_lyr_batch_prev_states = batch_prev_states[lyr]

      # Get the `lyr`-th RNN layer and the `lyr`-th dropout layer.
      rnn_lyr = self.stack_rnn[2 * lyr]
      dropout_lyr = self.stack_rnn[2 * lyr + 1]

      # Use previous RNN layer's output as next RNN layer's input.
      # Apply dropout to the output.
      # In  shape: (B, S, d_hid).
      # Out shape: (B, S, d_hid).
      rnn_lyr_out = rnn_lyr(batch_x=rnn_lyr_in, batch_prev_states=rnn_lyr_batch_prev_states)
      rnn_lyr_out = dropout_lyr(rnn_lyr_out)

      # Record the last hidden states.
      batch_cur_states.append(rnn_lyr_out[:, -1, :].detach())

      # Update RNN layer's input.
      rnn_lyr_in = rnn_lyr_out

    # Transform hidden states to next token embeddings.
    # Shape: (B, S, d_emb).
    z = self.fc_h2e(rnn_lyr_out)

    # Calculate similarity scores by calculating inner product over all token embeddings.
    # Shape: (B, S, V).
    sim = z @ self.emb.weight.transpose(0, 1)
    return (sim, batch_cur_states)

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
      There are ``n_lyr`` tensors in the list, the shape of each tensor is :math:`(B, \dHid)` and
      ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states of each layer.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the mini-batch cross-entropy loss with shape :math:`(1)` and
      ``dtype == torch.float``.
      The second item in the tuple is a list of tensor.
      Each tensor in the list is the last hiddent states of each recurrent layer derived from current input token ids.
      Each tensor has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
    """
    # Get next token id logits and the last hidden states.
    # Logits shape: (B, S, V).
    # Each tensor in `batch_cur_states` has shape: (B, d_hid).
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
      There are ``n_lyr`` tensors in the list, the shape of each tensor is :math:`(B, \dHid)` and
      ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states of each layer.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id probability distribution over the paired tokenizer's
      vocabulary.
      Probability tensor has shape :math:`(B, S, V)` and ``dtype == torch.float``.
      The second item in the tuple is a list of tensor.
      Each tensor in the list is the last hiddent states of each recurrent layer derived from current input token ids.
      Each tensor has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
    """
    # Get next token id logits and the last hidden states.
    # Logits shape: (B, S, V).
    # Each tensor in `batch_cur_states` has shape: (B, d_hid).
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate next token id probability distribution using softmax.
    # shape: (B, S, V).
    return (F.softmax(logits, dim=-1), batch_cur_states)
