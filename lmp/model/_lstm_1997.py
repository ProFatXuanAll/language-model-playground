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


class LSTM1997Layer(nn.Module):
  r"""LSTM (1997 version) [1]_ recurrent neural network.

  Implement RNN model in the paper `Long Short-Term Memory`_.

  .. _`Long Short-Term Memory`: https://ieeexplore.ieee.org/abstract/document/6795963

  Let :math:`\newcommand{\dBlk}{d_{\operatorname{blk}}} \dBlk` be the number of units in a memory cell block.
  Let :math:`\newcommand{\nBlk}{n_{\operatorname{blk}}} \nBlk` be the number of memory cell blocks.
  Let :math:`x` be input features with shape :math:`(B, S, \newcommand{\hIn}{H_{\operatorname{in}}} \hIn)`, where
  :math:`B` is batch size, :math:`S` is sequence length and :math:`\hIn` is the number of input features per time step
  in each sequence.
  Let :math:`h_0` be the initial hidden states with shape :math:`(B, \newcommand{\hOut}{H_{\operatorname{out}}} \hOut)`
  where :math:`\hOut = \nBlk \times \dBlk`.
  Let :math:`c_0` be the initial hidden states with shape :math:`(B, \nBlk, \dBlk)`.

  LSTM (1997 version) layer is defined as follow:

  .. math::

    \newcommand{\pa}[1]{\left( #1 \right)}
    \newcommand{\cat}[1]{\operatorname{concate}\pa{#1}}
    \newcommand{\eq}{\leftarrow}
    \newcommand{\fla}[1]{\operatorname{flatten}\pa{#1}}
    \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
    \begin{align*}
      & \textbf{procedure } \text{LSTM1997Layer}(x, [h_0, c_0])                                    \\
      & \hspace{1em} S \eq x.\text{size}(1)                                                        \\
      & \hspace{1em} \textbf{for } t \in \set{0, \dots, S-1} \textbf{ do}                          \\
      & \hspace{2em} i_t \eq \sigma(W_i \cdot x_t + U_i \cdot h_t + b_i)                           \\
      & \hspace{2em} o_t \eq \sigma(W_o \cdot x_t + U_o \cdot h_t + b_o)                           \\
      & \hspace{2em} \textbf{for } k \in \set{0, \dots, \nBlk-1} \textbf{ do}                      \\
      & \hspace{3em} g_{t,k} = \tanh\pa{W_k \cdot x_t + U_k \cdot h_t + b_k}    &&\tag{1}\label{1} \\
      & \hspace{3em} c_{t+1,k} = c_{t,k} + i_{t,k} \cdot g_{t,k}                                   \\
      & \hspace{3em} h_{t+1,k} = o_{t,k} \cdot \tanh(c_{t+1,k})                 &&\tag{2}\label{2} \\
      & \hspace{2em} \textbf{end for}                                                              \\
      & \hspace{2em} c_{t+1} \eq \cat{c_{t+1,0}, \dots, c_{t+1,\nBlk-1}}                           \\
      & \hspace{2em} h_{t+1} \eq \fla{h_{t+1,0}, \dots, h_{t+1,\nBlk-1}}                           \\
      & \hspace{1em} \textbf{end for}                                                              \\
      & \hspace{1em} c \eq \cat{c_1, \dots, c_S}                                                   \\
      & \hspace{1em} h \eq \cat{h_1, \dots, h_S}                                                   \\
      & \hspace{1em} \textbf{return } [h, c]                                                       \\
      & \textbf{end procedure}
    \end{align*}

  +-----------------------------------------+------------------------------------------------+
  | Trainable Parameters                    | Nodes                                          |
  +-------------+---------------------------+-----------------+------------------------------+
  | Parameter   | Shape                     | Symbol          | Shape                        |
  +=============+===========================+=================+==============================+
  | :math:`h_0` | :math:`(1, \hOut)`        | :math:`x`       | :math:`(B, S, \hIn)`         |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`c_0` | :math:`(1, \nBlk, \dBlk)` | :math:`h_0`     | :math:`(B, \hOut)`           |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`W_i` | :math:`(\nBlk, \hIn)`     | :math:`c_0`     | :math:`(B, \nBlk, \dBlk)`    |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`U_i` | :math:`(\nBlk, \hOut)`    | :math:`x_t`     | :math:`(B, \hIn)`            |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`b_i` | :math:`(\nBlk)`           | :math:`h_t`     | :math:`(B, \hOut)`           |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`W_o` | :math:`(\nBlk, \hIn)`     | :math:`i_t`     | :math:`(B, \nBlk)`           |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`U_o` | :math:`(\nBlk, \hOut)`    | :math:`o_t`     | :math:`(B, \nBlk)`           |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`b_o` | :math:`(\nBlk)`           | :math:`g_{t,k}` | :math:`(B, \dBlk)`           |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`W_k` | :math:`(\dBlk, \hIn)`     | :math:`c_{t,k}` | :math:`(B, \dBlk)`           |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`U_k` | :math:`(\dBlk, \hOut)`    | :math:`i_{t,k}` | :math:`(B, 1)`               |
  +-------------+---------------------------+-----------------+------------------------------+
  | :math:`b_k` | :math:`(\dBlk)`           | :math:`o_{t,k}` | :math:`(B, 1)`               |
  +-------------+---------------------------+-----------------+------------------------------+
  |                                         | :math:`h_{t,k}` | :math:`(B, \dBlk)`           |
  |                                         +-----------------+------------------------------+
  |                                         | :math:`c_t`     | :math:`(B, \nBlk, \dBlk)`    |
  |                                         +-----------------+------------------------------+
  |                                         | :math:`c`       | :math:`(B, S, \nBlk, \dBlk)` |
  |                                         +-----------------+------------------------------+
  |                                         | :math:`h`       | :math:`(B, S, \hOut)`        |
  +-----------------------------------------+-----------------+------------------------------+

  - :math:`i_t` is memory cell blocks' input gate units at time step :math:`t`.
    :math:`i_{t,k}` is the :math:`k`-th coordinates of :math:`i_t`, which represents the :math:`k`-th memory cell
    block's input gate unit at time step :math:`t`.
  - :math:`o_t` is memory cell blocks' output gate units at time step :math:`t`.
    :math:`o_{t,k}` is the :math:`k`-th coordinates of :math:`o_t`, which represents the :math:`k`-th memory cell
    block's output gate unit at time step :math:`t`.
  - The :math:`k`-th memory cell block is consist of:

    - Current input features :math:`x_t`.
    - Previous hidden states :math:`h_t`.
    - Input activations :math:`g_{t,k}`.
    - A input gate unit :math:`i_{t,k}`.
    - A output gate unit :math:`o_{t,k}`.
    - Previous internal states :math:`c_{t,k}` and current internal states :math:`c_{t+1,k}`.
    - Output units :math:`h_{t+1,k}`.

  - All memory cell blocks' current internal states at time step :math:`t` are concatenated to form the new internal
    states :math:`c_{t+1}`.
    The initial internal states :math:`c_0` is a trainable parameter.
  - All memory cell blocks' output units at time step :math:`t` are flattened to form the new hidden states
    :math:`h_{t+1}`.
    The initial hidden states :math:`h_0` is a trainable parameter.
  - Our implementation use :math:`\tanh` as memory cell blocks' input activation function.
    The implementation in the paper use :math:`4 \sigma - 2` in :math:`\eqref{1}` and :math:`2 \sigma - 1` in
    :math:`\eqref{2}`.
    We argue that the change in :math:`\eqref{1}` does not greatly affect the computation result and :math:`\eqref{2}`
    is almost the same as the paper implementation.
    To be precise, :math:`\tanh(x) = 2 \sigma(2x) - 1`.
    The formula :math:`2 \sigma(x) - 1` has gradient :math:`2 \sigma(x) (1 - \sigma(x))`.
    The formula :math:`\tanh(x)` has gradient :math:`4 \sigma(2x) (1 - \sigma(2x))`.
    Intuitively using :math:`\tanh` will scale gradient by 4.

  Parameters
  ----------
  d_blk: int
    Dimension of each memory cell block :math:`\dBlk`.
  in_feat: int
    Number of input features :math:`\hIn`.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  c_0: torch.nn.Parameter
    Memory cell blocks' initial internal states :math:`c_0`.
    Shape: :math:`(1, \nBlk, \dBlk)`.
  d_blk: int
    Dimension of each memory cell block :math:`\dBlk`.
  d_hid: int
    Total number of memory cell units :math:`\hOut`.
  fc_h2ig: torch.nn.Linear
    Fully connected layer :math:`U_i` which connects hidden states to memory cell's input gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  fc_h2mc_in: torch.nn.Linear
    Fully connected layers :math:`\pa{U_0, \dots, U_{\nBlk-1}}` which connect hidden states to memory cell
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
    Fully connected layers :math:`\pa{W_0, \dots, W_{\nBlk-1}}` and :math:`\pa{b_0, \dots, b_{\nBlk-1}}` which connects
    input units to memory cell blocks' input activations.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \dHid)`.
  fc_x2og: torch.nn.Linear
    Fully connected layer :math:`W_o` and :math:`b_o` which connects input units to memory cell's output gate units.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \nBlk)`.
  h_0: torch.nn.Parameter
    Initial hidden states :math:`h_0`.
    Shape: :math:`(1, \dHid)`
  in_feat: int
    Number of input features :math:`\hIn`.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.

  References
  ----------
  .. [1] S. Hochreiter and J. Schmidhuber, "`Long Short-Term Memory`_," in Neural Computation, vol. 9, no. 8,
     pp. 1735-1780, 15 Nov. 1997, doi: 10.1162/neco.1997.9.8.1735.
  """

  def __init__(self, d_blk: int, in_feat: int, n_blk: int, **kwargs: Any):
    super().__init__()

    # `d_blk` validation.
    lmp.util.validate.raise_if_not_instance(val=d_blk, val_name='d_blk', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, d_blk], val_names=['1', 'd_blk'])
    self.d_blk = d_blk

    # `in_feat` validation.
    lmp.util.validate.raise_if_not_instance(val=in_feat, val_name='in_feat', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, in_feat], val_names=['1', 'in_feat'])
    self.in_feat = in_feat

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

    # Initial hidden states and initial memory cell internal states.
    # First dimension is set to `1` to so that they can broadcast along batch dimension.
    self.h_0 = nn.Parameter(torch.zeros(1, self.d_hid))
    self.c_0 = nn.Parameter(torch.zeros(1, n_blk, d_blk))

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases other than :math:`b_i, b_o` are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, \dfrac{1}{\sqrt{d}}}` where :math:`d = \max(\hIn, \hOut)`.
    :math:`b_i, b_o` are initialized with uniform distribution :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, 0}` so that
    input and output gates remain closed at the begining of training.

    Returns
    -------
    None
    """
    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(max(self.in_feat, self.d_hid))

    nn.init.uniform_(self.fc_x2ig.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_x2og.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2ig.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2og.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_x2mc_in.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_x2mc_in.bias, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2mc_in.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.h_0, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.c_0, -inv_sqrt_dim, inv_sqrt_dim)

    # Gate units' biases are initialized to negative values.
    nn.init.uniform_(self.fc_x2ig.bias, -inv_sqrt_dim, 0.0)
    nn.init.uniform_(self.fc_x2og.bias, -inv_sqrt_dim, 0.0)

  def forward(
    self,
    batch_x: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> List[torch.Tensor]:
    r"""Calculate batch of hidden states for ``batch_x``.

    Below we describe the forward pass algorithm of LSTM (1997 version) layer.

    #. Let ``batch_x`` be batch input features :math:`x`.
    #. Let ``batch_prev_states`` be the initial hidden states :math:`h_0` and the initial internal states :math:`c_0`.
       If ``batch_prev_states is None``, use ``self.h_0`` and ``self.c_0`` instead.
    #. Let ``batch_x.size(1)`` be sequence length :math:`S`.
    #. Loop through :math:`\set{0, \dots, S-1}` with looping index :math:`t`.

       #. Use :math:`x_t`, :math:`h_t`, ``self.fc_x2ig`` and ``self.fc_h2ig`` to get input gate units :math:`i_t`.
       #. Use :math:`x_t`, :math:`h_t`, ``self.fc_x2og`` and ``self.fc_h2og`` to get output gate units :math:`o_t`.
       #. Use :math:`x_t`, :math:`h_t`, ``self.fc_x2mc_in`` and ``self.fc_h2mc_in`` to get memory cell input
          activations :math:`g_{t,0}, \dots, g_{t,\nBlk-1}`.
       #. Derive new internal state :math:`c_{t+1}` using input gate units :math:`i_{t,0}, \dots, i_{t,\nBlk-1}` and
          memory cell input activations :math:`g_{t,0}, \dots, g_{t,\nBlk-1}`.
       #. Derive new hidden states :math:`h_{t+1}` using output gate units :math:`o_{t,0}, \dots, o_{t,\nBlk-1}` and
          new internal states :math:`c_{t+1}`.

    #. Denote the concatenation of internal states :math:`c_1, \dots, c_S` as :math:`c`.
    #. Denote the concatenation of hidden states :math:`h_1, \dots, h_S` as :math:`h`.
    #. Return :math:`(h, c)`.

    Parameters
    ----------
    batch_x: torch.Tensor
      Batch input features.
      ``batch_x`` has shape :math:`(B, S, \hIn)` and ``dtype == torch.float``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Two items list containing batch previous hidden states and batch previous internal states.
      ``batch_prev_states[0]`` has shape :math:`(B, \hOut)` and ``dtype == torch.float``.
      ``batch_prev_states[0]`` has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states :math:`h_0` and the initial internal state :math:`c_0`.

    Returns
    -------
    list[torch.Tensor]
      Two items list containing batch hidden states and batch internal states.
      Batch hidden states has shape :math:`(B, S, \hOut)` and ``dtype == torch.float``.
      Batch internal states has shape :math:`(B, S, \nBlk, \dBlk)` and ``dtype == torch.float``.
    """
    if batch_prev_states is None:
      batch_prev_states = [self.h_0, self.c_0]

    h_prev = batch_prev_states[0]
    c_prev = batch_prev_states[1]

    # Sequence length.
    S = batch_x.size(1)

    # Transform input features to gate units.
    # Shape: (B, S, n_blk).
    x2ig = self.fc_x2ig(batch_x)
    x2og = self.fc_x2og(batch_x)

    # Transform input features to memory cell block's input.
    # Shape: (B, S, d_hid).
    x2mc_in = self.fc_x2mc_in(batch_x)

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

      # Calculate memory cell blocks' new internal states.
      # Shape: (B, n_blk, d_blk).
      c_cur = c_prev + ig * mc_in

      # Calculate memory cell blocks' outputs and flatten to form the new hidden states.
      # Shape: (B, d_hid).
      h_cur = (og * torch.tanh(c_cur)).reshape(-1, self.d_hid)

      c_all.append(c_cur)
      h_all.append(h_cur)

      # Update hidden states and memory cell blocks' internal states.
      c_prev = c_cur
      h_prev = h_cur

    # Stack list of tensors into single tensor.
    # In  shape: list of (B, d_hid) with length equals to `S`.
    # Out shape: (B, S, d_hid).
    h = torch.stack(h_all, dim=1)

    # Stack list of tensors into single tensor.
    # In  shape: list of (B, n_blk, d_blk) with length equals to `S`.
    # Out shape: (B, S, n_blk, d_blk).
    c = torch.stack(c_all, dim=1)

    return [h, c]


class LSTM1997(BaseModel):
  r"""LSTM (1997 version) language model.

  Implement RNN model in the paper `Long Short-Term Memory`_ as a language model.

  .. _`Long Short-Term Memory`: https://ieeexplore.ieee.org/abstract/document/6795963

  - Let :math:`x` be batch of token ids with shape :math:`(B, S)`, where :math:`B` is batch size and :math:`S` is
    sequence length.
  - Let :math:`V` be the vocabulary size of the paired tokenizer.
    Each token id represents an unique token, i.e., :math:`x_t \in \set{0, \dots, V -1}`.
  - Let :math:`E` be the token embedding lookup table.

    - Let :math:`\newcommand{\dEmb}{d_{\operatorname{emb}}} \dEmb` be the dimension of token embeddings.
    - Let :math:`e_t` be the token embedding correspond to token id :math:`x_t`.

  - Let :math:`\newcommand{\nLyr}{n_{\operatorname{lyr}}} \nLyr` be the number of recurrent layers.
  - Let :math:`\newcommand{\dBlk}{d_{\operatorname{blk}}} \dBlk` be the number of units in a memory cell block.
  - Let :math:`\newcommand{\nBlk}{n_{\operatorname{blk}}} \nBlk` be the number of memory cell blocks.
  - Let :math:`\newcommand{\dHid}{d_{\operatorname{hid}}} \dHid = \nBlk \times \dBlk`.
  - Let :math:`h^\ell` be the hidden states of the :math:`\ell` th recurrent layer, let :math:`h_t^\ell` be the
    :math:`t` th time step of :math:`h^\ell` and let :math:`h_0^\ell` be the initial hidden states of the :math:`\ell`
    th recurrent layer.
    The initial hidden states are given as input.
  - Let :math:`c^\ell` be the internal states of the :math:`\ell` th recurrent layer, let :math:`c_t^\ell` be the
    :math:`t` th time step of :math:`c^\ell` and let :math:`c_0^\ell` be the initial internal states of the
    :math:`\ell` th recurrent layer.
    The initial internal states are given as input.

  LSTM (1997 version) language model is defined as follow:

  .. math::

    \newcommand{\br}[1]{\left[ #1 \right]}
    \newcommand{\eq}{\leftarrow}
    \newcommand{\pa}[1]{\left( #1 \right)}
    \newcommand{\cat}[1]{\operatorname{concate}\pa{#1}}
    \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
    \begin{align*}
      & \textbf{procedure }\text{LSTM1997}\pa{x, \br{h_0^0, c_0^0, \dots, h_0^{\nLyr-1}, c_0^{\nLyr-1}}}              \\
      & \hspace{1em} \textbf{for } t \in \set{0, \dots, S-1} \textbf{ do}                                             \\
      & \hspace{2em} e_t \eq (x_t)\text{-th row of } E \text{ but treated as column vector}                           \\
      & \hspace{2em} h_t^{-1} \eq \tanh(W_h \cdot e_t + b_h)                                                          \\
      & \hspace{1em} \textbf{end for}                                                                                 \\
      & \hspace{1em} h^{-1} \eq \cat{h_0^{-1}, \dots, h_{S-1}^{-1}}                                                   \\
      & \hspace{1em} \textbf{for } \ell \in \set{0, \dots, \nLyr-1} \textbf{ do}                                      \\
      & \hspace{2em} [h^\ell, c^\ell] \eq \text{LSTM1997Layer}(x \eq h^{\ell-1}, [h_0, c_0] \eq [h_0^\ell, c_0^\ell]) \\
      & \hspace{1em} \textbf{end for}                                                                                 \\
      & \hspace{1em} \textbf{for } t \in \set{0, \dots, S-1} \textbf{ do}                                             \\
      & \hspace{2em} z_{t+1} \eq \tanh\pa{W_z \cdot h_{t+1}^{\nLyr-1} + b_z}                                          \\
      & \hspace{2em} y_{t+1} \eq \sof{E \cdot z_{t+1}}                                                                \\
      & \hspace{1em} \textbf{end for}                                                                                 \\
      & \hspace{1em} y \eq \cat{y_1, \dots, y_S}                                                                      \\
      & \hspace{1em} \textbf{return } \pa{y, \br{h_S^0, c_S^0, \dots, h_S^{\nLyr-1}, c_S^{\nLyr-1}}}                  \\
      & \textbf{end procedure}
    \end{align*}

  +----------------------------------------------+-------------------------------------------------+
  | Trainable Parameters                         | Nodes                                           |
  +------------------+---------------------------+------------------+------------------------------+
  | Parameter        | Shape                     | Symbol           | Shape                        |
  +==================+===========================+==================+==============================+
  | :math:`h_0^\ell` | :math:`(1, \dHid)`        | :math:`x_t`      | :math:`(B, S)`               |
  +------------------+---------------------------+------------------+------------------------------+
  | :math:`c_0^\ell` | :math:`(1, \nBlk, \dBlk)` | :math:`c_0^\ell` | :math:`(B, \nBlk, \dBlk)`    |
  +------------------+---------------------------+------------------+------------------------------+
  | :math:`E`        | :math:`(V, \dEmb)`        | :math:`h_0^\ell` | :math:`(B, \dHid)`           |
  +------------------+---------------------------+------------------+------------------------------+
  | :math:`W_h`      | :math:`(\dHid, \dEmb)`    | :math:`e_t`      | :math:`(B, S, \dEmb)`        |
  +------------------+---------------------------+------------------+------------------------------+
  | :math:`b_h`      | :math:`(\dHid)`           | :math:`h_t^{-1}` | :math:`(B, \dHid)`           |
  +------------------+---------------------------+------------------+------------------------------+
  | :math:`W_z`      | :math:`(\dEmb, \dHid)`    | :math:`h^{-1}`   | :math:`(B, S, \dHid)`        |
  +------------------+---------------------------+------------------+------------------------------+
  | :math:`b_z`      | :math:`(\dEmb)`           | :math:`h^\ell`   | :math:`(B, S, \dHid)`        |
  +------------------+---------------------------+------------------+------------------------------+
  | :math:`\text{LSTM1997Layer}`                 | :math:`h_t^\ell` | :math:`(B, \dHid)`           |
  +----------------------------------------------+------------------+------------------------------+
  |                                              | :math:`c^\ell`   | :math:`(B, S, \nBlk, \dBlk)` |
  |                                              +------------------+------------------------------+
  |                                              | :math:`c_t^\ell` | :math:`(B, \nBlk, \dBlk)`    |
  |                                              +------------------+------------------------------+
  |                                              | :math:`z_t`      | :math:`(B, \dEmb)`           |
  |                                              +------------------+------------------------------+
  |                                              | :math:`y_t`      | :math:`(B, V)`               |
  |                                              +------------------+------------------------------+
  |                                              | :math:`y`        | :math:`(B, S, V)`            |
  +----------------------------------------------+------------------+------------------------------+

  - :math:`z_{t+1}` is obtained by transforming :math:`h_{t+1}^{\nLyr-1}` from dimension :math:`\dHid` to :math:`\dEmb`.
    This is only need for shape consistency:
    the hidden states :math:`h_{t+1}^{\nLyr-1}` has shape :math:`(B, \dHid)`, and the final output :math:`y_{t+1}` has
    shape :math:`(B, V)`.
  - :math:`y_{t+1}` is the next token id prediction probability distribution over tokenizer's vocabulary.
    We use inner product to calculate similarity scores over all token ids, and then use softmax to normalize
    similarity scores into probability range :math:`[0, 1]`.
  - The calculations after hidden states are the same as :py:class:`lmp.model.ElmanNet`.

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
  d_blk: int
    Dimension of each memory cell block :math:`\dBlk`.
  d_hid: int
    Total number of memory cell units :math:`\dHid`.
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
    CLI name of LSTM (1997 version) is ``LSTM-1997``.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.
  n_lyr: int
    Number of recurrent layers :math:`\nLyr`.
  p_emb: float
    Embeddings dropout probability.
  p_hid: float
    Hidden units dropout probability.
  stack_rnn: torch.nn.ModuleList
    :py:class:`lmp.model.LSTM1997Layer` stacking layers.
    Each LSTM (1997 version) layer is followed by a dropout layer with probability ``p_hid``.
    The number of stacking layers is equal to ``2 * n_lyr``.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dHid)`.

  See Also
  --------
  lmp.model.ElmanNet
    Elman Net language model.
  lmp.model.LSTM1997Layer
    LSTM (1997 version) recurrent neural network.
  """

  model_name: ClassVar[str] = 'LSTM-1997'

  def __init__(
    self,
    *,
    d_blk: int,
    d_emb: int,
    n_blk: int,
    n_lyr: int,
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
    self.d_emb = d_emb

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

    # Fully connected layer which connects input units to the 0th recurrent layer's input.
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
      self.stack_rnn.append(LSTM1997Layer(d_blk=d_blk, in_feat=self.d_hid, n_blk=n_blk))
      self.stack_rnn.append(nn.Dropout(p=p_hid))

    # Fully connected layer which transforms hidden states to next token embeddings.
    # Dropout is applied to make transform robust.
    self.fc_h2e = nn.Sequential(
      nn.Linear(in_features=self.d_hid, out_features=d_emb),
      nn.Tanh(),
      nn.Dropout(p=p_hid),
    )

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases other than :py:class:`lmp.model.LSTM1997Layer` are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, \dfrac{1}{\sqrt{d}}}` where :math:`d = \max(\dEmb, \dBlk \times \nBlk)`.
    :py:class:`lmp.model.LSTM1997Layer` are initialized by :py:meth:`lmp.model.LSTM1997Layer.params_init`.

    Returns
    -------
    None

    See Also
    --------
    lmp.model.LSTM1997Layer.params_init
      LSTM (1997 version) layer parameter initialization.
    """
    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(max(self.emb.embedding_dim, self.d_hid))

    nn.init.uniform_(self.emb.weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2h[1].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_e2h[1].bias, -inv_sqrt_dim, inv_sqrt_dim)
    for lyr in range(self.n_lyr):
      self.stack_rnn[2 * lyr].params_init()
    nn.init.uniform_(self.fc_h2e[0].weight, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.fc_h2e[0].bias, -inv_sqrt_dim, inv_sqrt_dim)

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
    lmp.script.train_model
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
    ...   '--n_lyr', '2',
    ...   '--p_emb', '0.5',
    ...   '--p_hid', '0.1',
    ... ])
    >>> assert args.d_blk == 64
    >>> assert args.d_emb == 100
    >>> assert args.n_blk == 8
    >>> assert args.n_lyr == 2
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
      '--n_lyr',
      help='Number of LSTM (1997 version) layers.',
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
      Batch current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states and batch of internal states.
      There are :math:`2 \nLyr` tensors in the list.
      The first and the second tensors are the 0th layer's batch of previous hidden states and batch of previous
      internal states, the third and the fourth tensors are the 1st layer's batch of previous hidden states and batch
      of previous internal states, and so on.
      Batch previous hidden states has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Batch previous internal states has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states and initial internal states of each layer.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id logits with shape :math:`(B, S, V)` and
      ``dtype == torch.float``.
      The second item in the tuple is a list of tensor represents the last hidden states and the last internal states
      of each layer.
      The structure of the tensor list is the same as ``batch_prev_states``.
    """
    # Use initial hidden states if `batch_prev_states is None`.
    if batch_prev_states is None:
      batch_prev_states = [None] * (2 * self.n_lyr)

    # Lookup token embeddings and feed to recurrent units.
    # In  shape: (B, S).
    # Out shape: (B, S, d_hid).
    rnn_lyr_in = self.fc_e2h(self.emb(batch_cur_tkids))

    # Loop through each layer and gather the last hidden states of each layer.
    batch_cur_states = []
    for lyr in range(self.n_lyr):
      # Fetch previous hidden state of a layer.
      # Shape: (B, S, d_hid).
      rnn_lyr_batch_h_prev = batch_prev_states[2 * lyr]

      # Fetch previous internal state of a layer.
      # Shape: (B, S, n_blk, d_blk).
      rnn_lyr_batch_c_prev = batch_prev_states[2 * lyr + 1]

      if rnn_lyr_batch_h_prev is None or rnn_lyr_batch_c_prev is None:
        rnn_lyr_batch_prev_states = None
      else:
        rnn_lyr_batch_prev_states = [rnn_lyr_batch_h_prev, rnn_lyr_batch_c_prev]

      # Get the `lyr`-th RNN layer and the `lyr`-th dropout layer.
      rnn_lyr = self.stack_rnn[2 * lyr]
      dropout_lyr = self.stack_rnn[2 * lyr + 1]

      # Use previous RNN layer's output as next RNN layer's input.
      # Apply dropout to the output.
      # In  shape: (B, S, d_hid).
      # rnn_lyr_h_out shape: (B, S, d_hid).
      rnn_lyr_h_out, rnn_lyr_c_out = rnn_lyr(batch_x=rnn_lyr_in, batch_prev_states=rnn_lyr_batch_prev_states)
      rnn_lyr_h_out = dropout_lyr(rnn_lyr_h_out)

      # Record the last hidden states.
      batch_cur_states.append(rnn_lyr_h_out[:, -1, :].detach())

      # Record the last internal states.
      batch_cur_states.append(rnn_lyr_c_out[:, -1, :, :].detach())

      # Update RNN layer's input.
      rnn_lyr_in = rnn_lyr_h_out

    # Transform hidden states to next token embeddings.
    # Shape: (B, S, d_emb).
    z = self.fc_h2e(rnn_lyr_h_out)

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

    We use cross entropy loss as our training objective.
    This method must only be used for training.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_next_tkids: torch.Tensor
      Ground truth of each sample in the batch.
      ``batch_next_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states and batch of internal states.
      There are :math:`2 \nLyr` tensors in the list.
      The first and the second tensors are the 0th layer's batch of previous hidden states and batch of previous
      internal states, the third and the fourth tensors are the 1st layer's batch of previous hidden states and batch
      of previous internal states, and so on.
      Batch previous hidden states has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Batch previous internal states has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states and initial internal states of each layer.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the mini-batch cross-entropy loss with shape :math:`(1)` and
      ``dtype == torch.float``.
      The second item in the tuple is a list of tensor represents the last hidden states and the last internal states
      of each layer.
      The structure of the tensor list is the same as ``batch_prev_states``.
    """
    # Get next token id logits and the last hidden states.
    # Logits shape: (B, S, V)
    # Last hidden states and internal states shapes: [(B, d_hid), (B, n_blk, d_blk)]
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
      Batch current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous hidden states and batch of internal states.
      There are :math:`2 \nLyr` tensors in the list.
      The first and the second tensors are the 0th layer's batch of previous hidden states and batch of previous
      internal states, the third and the fourth tensors are the 1st layer's batch of previous hidden states and batch
      of previous internal states, and so on.
      Batch previous hidden states has shape :math:`(B, \dHid)` and ``dtype == torch.float``.
      Batch previous internal states has shape :math:`(B, \nBlk, \dBlk)` and ``dtype == torch.float``.
      Set to ``None`` to use the initial hidden states and initial internal states of each layer.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id probability distribution over the tokenizer's
      vocabulary.
      Probability tensor has shape :math:`(B, S, V)` and ``dtype == torch.float``.
      The second item in the tuple is a list of tensor represents the last hidden states and the last internal states
      of each layer.
      The structure of the tensor list is the same as ``batch_prev_states``.
    """
    # Get next token id logits and the last hidden states.
    # Logits shape: (B, S, V)
    # Last hidden states and internal states shapes: [(B, d_hid), (B, n_blk, d_blk)]
    logits, batch_cur_states = self(batch_cur_tkids=batch_cur_tkids, batch_prev_states=batch_prev_states)

    # Calculate next token id probability distribution using softmax.
    # shape: (B, S, V).
    return (F.softmax(logits, dim=-1), batch_cur_states)
