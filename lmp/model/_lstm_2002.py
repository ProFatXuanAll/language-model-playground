"""LSTM (2002 version) language model."""

import argparse
import math
from typing import Any, ClassVar, List, Optional

import torch
import torch.nn as nn

import lmp.util.metric
import lmp.util.validate
from lmp.model._lstm_2000 import LSTM2000, LSTM2000Layer
from lmp.tknzr._base import BaseTknzr


class LSTM2002Layer(LSTM2000Layer):
  r"""LSTM (2002 version) [1]_ recurrent neural network.

  Implement RNN model in the paper `Learning Precise Timing with LSTM Recurrent Networks`_.

  .. _`Learning Precise Timing with LSTM Recurrent Networks`: https://www.jmlr.org/papers/v3/gers02a.html

  Let :math:`\newcommand{\dBlk}{d_{\operatorname{blk}}} \dBlk` be the number of units in a memory cell block.
  Let :math:`\newcommand{\nBlk}{n_{\operatorname{blk}}} \nBlk` be the number of memory cell blocks.
  Let :math:`x` be input features with shape :math:`(B, S, \newcommand{\hIn}{H_{\operatorname{in}}} \hIn)`, where
  :math:`B` is batch size, :math:`S` is sequence length and :math:`\hIn` is the number of input features per time step
  in each sequence.
  Let :math:`h_0` be the initial hidden states with shape :math:`(B, \newcommand{\hOut}{H_{\operatorname{out}}} \hOut)`
  where :math:`\hOut = \nBlk \times \dBlk`.
  Let :math:`c_0` be the initial hidden states with shape :math:`(B, \nBlk, \dBlk)`.

  LSTM (2002 version) layer is defined as follow:

  .. math::

    \newcommand{\pa}[1]{\left( #1 \right)}
    \newcommand{\cat}[1]{\operatorname{concate}\pa{#1}}
    \newcommand{\eq}{\leftarrow}
    \newcommand{\fla}[1]{\operatorname{flatten}\pa{#1}}
    \begin{align*}
      & \textbf{procedure } \text{LSTM2002Layer}(x, [h_0, c_0])                                                    \\
      & \hspace{1em} S \eq x.\text{size}(1)                                                                        \\
      & \hspace{1em} \textbf{for } t \in \set{0, \dots, S-1} \textbf{ do}                                          \\
      & \hspace{2em} \textbf{for } k \in \set{0, \dots, \nBlk-1} \textbf{ do}                                      \\
      & \hspace{3em} f_{t,k} \eq \sigma(W_{f,k} \cdot x_t + U_{f,k} \cdot h_t + V_{f,k} \cdot c_{t,k} + b_{f,k})
        &&\tag{1}\label{1}                                                                                         \\
      & \hspace{3em} i_{t,k} \eq \sigma(W_{i,k} \cdot x_t + U_{i,k} \cdot h_t + V_{i,k} \cdot c_{t,k} + b_{i,k})
        &&\tag{2}\label{2}                                                                                         \\
      & \hspace{3em} g_{t,k} = \tanh\pa{W_k \cdot x_t + U_k \cdot h_t + b_k}
        &&\tag{3}\label{3}                                                                                         \\
      & \hspace{3em} c_{t+1,k} = f_{t, k} \cdot c_{t,k} + i_{t,k} \cdot g_{t,k}                                    \\
      & \hspace{3em} o_{t,k} \eq \sigma(W_{o,k} \cdot x_t + U_{o,k} \cdot h_t + V_{o,k} \cdot c_{t+1,k} + b_{o,k})
        &&\tag{4}\label{4}                                                                                         \\
      & \hspace{3em} h_{t+1,k} = o_{t,k} \cdot \tanh(c_{t+1,k})
        &&\tag{5}\label{5}                                                                                         \\
      & \hspace{2em} \textbf{end for}                                                                              \\
      & \hspace{2em} c_{t+1} \eq \cat{c_{t+1,0}, \dots, c_{t+1,\nBlk-1}}                                           \\
      & \hspace{2em} h_{t+1} \eq \fla{h_{t+1,0}, \dots, h_{t+1,\nBlk-1}}                                           \\
      & \hspace{1em} \textbf{end for}                                                                              \\
      & \hspace{1em} c \eq \cat{c_1, \dots, c_S}                                                                   \\
      & \hspace{1em} h \eq \cat{h_1, \dots, h_S}                                                                   \\
      & \hspace{1em} \textbf{return } [h, c]                                                                       \\
      & \textbf{end procedure}
    \end{align*}

  +---------------------------------------------+------------------------------------------------+
  | Trainable Parameters                        | Nodes                                          |
  +-----------------+---------------------------+-----------------+------------------------------+
  | Parameter       | Shape                     | Symbol          | Shape                        |
  +=================+===========================+=================+==============================+
  | :math:`h_0`     | :math:`(1, \hOut)`        | :math:`x`       | :math:`(B, S, \hIn)`         |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`c_0`     | :math:`(1, \nBlk, \dBlk)` | :math:`h_0`     | :math:`(B, \hOut)`           |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`W_{f,k}` | :math:`(1, \hIn)`         | :math:`c_0`     | :math:`(B, \nBlk, \dBlk)`    |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`U_{f,k}` | :math:`(1, \hOut)`        | :math:`x_t`     | :math:`(B, \hIn)`            |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`V_{f,k}` | :math:`(1, \dBlk)`        | :math:`h_t`     | :math:`(B, \hOut)`           |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`b_{f,k}` | :math:`(\nBlk)`           | :math:`c_{t,k}` | :math:`(B, \dBlk)`           |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`W_{i,k}` | :math:`(1, \hIn)`         | :math:`f_{t,k}` | :math:`(B, 1)`               |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`U_{i,k}` | :math:`(1, \hOut)`        | :math:`i_{t,k}` | :math:`(B, 1)`               |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`V_{i,k}` | :math:`(1, \dBlk)`        | :math:`g_{t,k}` | :math:`(B, \dBlk)`           |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`b_{i,k}` | :math:`(\nBlk)`           | :math:`o_{t,k}` | :math:`(B, 1)`               |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`W_{o,k}` | :math:`(1, \hIn)`         | :math:`h_{t,k}` | :math:`(B, \dBlk)`           |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`U_{o,k}` | :math:`(1, \hOut)`        | :math:`c_t`     | :math:`(B, \nBlk, \dBlk)`    |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`V_{o,k}` | :math:`(1, \dBlk)`        | :math:`c`       | :math:`(B, S, \nBlk, \dBlk)` |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`b_{o,k}` | :math:`(\nBlk)`           | :math:`h`       | :math:`(B, S, \hOut)`        |
  +-----------------+---------------------------+-----------------+------------------------------+
  | :math:`W_k`     | :math:`(\hIn, \dBlk)`     |                                                |
  +-----------------+---------------------------+                                                |
  | :math:`U_k`     | :math:`(\hOut, \dBlk)`    |                                                |
  +-----------------+---------------------------+                                                |
  | :math:`b_k`     | :math:`(\dBlk)`           |                                                |
  +-----------------+---------------------------+------------------------------------------------+

  - The differences between :py:class:`lmp.model.LSTM2000Layer` and :py:class:`lmp.model.LSTM2002Layer` are list below:

    - Input, forget and output gate units have peephole connections connect to memory cell blocks' internal states.
      See :math:`\eqref{eq1}\eqref{eq2}\eqref{eq4}`.
    - Output gate units can be calculated only after updating memory cell blocks' internal states.
      See :math:`\eqref{eq4}`.

  - The implementation in the paper use identity mappings in :math:`\eqref{eq3}\eqref{eq5}`.
    Our implementation use :math:`\tanh` instead.
    We argue that the changes in :math:`\eqref{eq3}\eqref{eq5}` make sure our model activations are bounded and the
    paper implementation is unbounded.
    Since one usually use much larger dimension to train language model compare to the paper (which use dimension
    :math:`1` on everything), activations of LSTM tend to grow to extremely positive / negative values without
    :math:`\tanh`.

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
  fc_h2fg: torch.nn.Linear
    Fully connected layers :math:`\pa{U_{f,0}, \dots, U_{f,\nBlk-1}}` which connect hidden states to memory cell's
    forget gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  fc_h2ig: torch.nn.Linear
    Fully connected layers :math:`\pa{U_{i,0}, \dots, U_{i,\nBlk-1}}` which connect hidden states to memory cell's
    input gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  fc_h2mc_in: torch.nn.Linear
    Fully connected layers :math:`\pa{U_0, \dots, U_{\nBlk-1}}` which connect hidden states to memory cell blocks'
    input activations.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \dHid)`.
  fc_h2og: torch.nn.Linear
    Fully connected layers :math:`\pa{U_{o,0}, \dots, U_{o,\nBlk-1}}` which connect hidden states to memory cell's
    output gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
  fc_x2fg: torch.nn.Linear
    Fully connected layers :math:`\pa{W_{f,0}, \dots, W_{f,\nBlk-1}}` and :math:`\pa{b_{f,0}, \dots, b_{f,\nBlk-1}}`
    which connect input units to memory cell's forget gate units.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \nBlk)`.
  fc_x2ig: torch.nn.Linear
    Fully connected layers :math:`\pa{W_{i,0}, \dots, W_{i,\nBlk-1}}` and :math:`\pa{b_{i,0}, \dots, b_{i,\nBlk-1}}`
    which connect input units to memory cell's input gate units.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \nBlk)`.
  fc_x2mc_in: torch.nn.Linear
    Fully connected layers :math:`\pa{W_0, \dots, W_{\nBlk-1}}` and :math:`\pa{b_0, \dots, b_{\nBlk-1}}` which connect
    input units to memory cell blocks' input activations.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \dHid)`.
  fc_x2og: torch.nn.Linear
    Fully connected layers :math:`\pa{W_{o,0}, \dots, W_{o,\nBlk-1}}` and :math:`\pa{b_{o,0}, \dots, b_{o,\nBlk-1}}`
    which connect input units to memory cell's output gate units.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \nBlk)`.
  h_0: torch.nn.Parameter
    Initial hidden states :math:`h_0`.
    Shape: :math:`(1, \dHid)`
  in_feat: int
    Number of input features :math:`\hIn`.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.
  pc_c2fg: torch.nn.Parameter
    Peephole connections :math:`\pa{V_{f,0}, \dots, V_{f,\nBlk-1}}` which connect the :math:`k`-th memory cell
    blocks' internal states to the :math:`k`-th forget gate units.
    Shape: :math:`(1, \nBlk, \dBlk)`.
  pc_c2ig: torch.nn.Parameter
    Peephole connections :math:`\pa{V_{i,0}, \dots, V_{i,\nBlk-1}}` which connect the :math:`k`-th memory cell
    blocks' internal states to the :math:`k`-th input gate units.
    Shape: :math:`(1, \nBlk, \dBlk)`.
  pc_c2og: torch.nn.Parameter
    Peephole connections :math:`\pa{V_{o,0}, \dots, V_{o,\nBlk-1}}` which connect the :math:`k`-th memory cell
    blocks' internal states to the :math:`k`-th output gate units.
    Shape: :math:`(1, \nBlk, \dBlk)`.

  See Also
  --------
  lmp.model.LSTM2000Layer
    LSTM (2000 version) recurrent neural network.

  References
  ----------
  .. [1] Gers, F. A., Schraudolph, N. N., & Schmidhuber, J. (2002). `Learning precise timing with LSTM recurrent
         networks`_. Journal of machine learning research, 3(Aug), 115-143.
  """

  def __init__(self, d_blk: int, in_feat: int, n_blk: int, **kwargs: Any):
    super().__init__(
      d_blk=d_blk,
      in_feat=in_feat,
      n_blk=n_blk,
      **kwargs,
    )

    # Peephole connections for gate units.
    # First dimension is set to `1` to broadcast along batch dimension.
    self.pc_c2fg = nn.Parameter(torch.zeros(1, n_blk, d_blk))
    self.pc_c2ig = nn.Parameter(torch.zeros(1, n_blk, d_blk))
    self.pc_c2og = nn.Parameter(torch.zeros(1, n_blk, d_blk))

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases other than :math:`b_{f,k}, b_{i,k}, b_{o,k}` are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, \dfrac{1}{\sqrt{d}}}` where :math:`d = \max(\hIn, \hOut)`.
    :math:`b_{i,k}, b_{o,k}` are initialized with uniform distribution :math:`\mathcal{U}\pa{\dfrac{-1}{\sqrt{d}}, 0}`
    so that input and output gates remain closed at the begining of training.
    :math:`b_{f,k}` is initialized with uniform distribution :math:`\mathcal{U}\pa{0, \dfrac{1}{\sqrt{d}}}` so that
    forget gates remain open at the begining of training.

    Returns
    -------
    None
    """
    super().params_init()

    # Initialize weights and biases with uniform distribution.
    inv_sqrt_dim = 1 / math.sqrt(max(self.in_feat, self.d_hid))

    nn.init.uniform_(self.pc_c2fg, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.pc_c2ig, -inv_sqrt_dim, inv_sqrt_dim)
    nn.init.uniform_(self.pc_c2og, -inv_sqrt_dim, inv_sqrt_dim)

  def forward(
    self,
    batch_x: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> List[torch.Tensor]:
    r"""Calculate batch of hidden states for ``batch_x``.

    Below we describe the forward pass algorithm of LSTM (2002 version) layer.

    #. Let ``batch_x`` be batch input features :math:`x`.
    #. Let ``batch_prev_states`` be the initial hidden states :math:`h_0` and the initial internal states :math:`c_0`.
       If ``batch_prev_states is None``, use ``self.h_0`` and ``self.c_0`` instead.
    #. Let ``batch_x.size(1)`` be sequence length :math:`S`.
    #. Loop through :math:`\set{0, \dots, S-1}` with looping index :math:`t`.

       #. Use :math:`x_t`, :math:`h_t`, :math:`c_t`, ``self.fc_x2fg``, ``self.fc_h2fg`` and ``self.pc_c2fg`` to get
          forget gate units :math:`f_{t,0}, \dots, f_{t,\nBlk-1}`.
       #. Use :math:`x_t`, :math:`h_t`, :math:`c_t`, ``self.fc_x2ig``, ``self.fc_h2ig`` and ``self.pc_c2ig`` to get
          input gate units :math:`i_{t,0}, \dots, i_{t,\nBlk-1}`.
       #. Use :math:`x_t`, :math:`h_t`, ``self.fc_x2mc_in`` and ``self.fc_h2mc_in`` to get memory cell input
          activations :math:`g_{t,0}, \dots, g_{t,\nBlk-1}`.
       #. Derive new internal state :math:`c_{t+1}` using forget gates units :math:`f_{t,0}, \dots, f_{t,\nBlk-1}`,
          input gate units :math:`i_{t,0}, \dots, i_{t,\nBlk-1}` and memory cell input activations
          :math:`g_{t,0}, \dots, g_{t,\nBlk-1}`.
       #. Use :math:`x_t`, :math:`h_t`, :math:`c_{t+1}`, ``self.fc_x2og``, ``self.fc_h2og`` and ``self.pc_c2og`` to get
          output gate units :math:`o_{t,0}, \dots, o_{t,\nBlk-1}`.
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
    x2fg = self.fc_x2fg(batch_x)
    x2ig = self.fc_x2ig(batch_x)
    x2og = self.fc_x2og(batch_x)

    # Transform input features to memory cell block's input.
    # Shape: (B, S, d_hid).
    x2mc_in = self.fc_x2mc_in(batch_x)

    # Perform recurrent calculation for `S` steps.
    c_all = []
    h_all = []
    for t in range(S):
      # Transform hidden states to forget / input / output gate units.
      # shape: (B, n_blk).
      h2fg = self.fc_h2fg(h_prev)
      h2ig = self.fc_h2ig(h_prev)
      h2og = self.fc_h2og(h_prev)

      # Calculate forget gate and input gate units peephole connections.
      # shape: (B, n_blk).
      c2fg = (self.pc_c2fg * c_prev).sum(dim=-1)
      c2ig = (self.pc_c2ig * c_prev).sum(dim=-1)

      # Get forget gate and input gate units and unsqueeze to separate memory cell blocks.
      # shape: (B, n_blk, 1).
      fg = torch.sigmoid(x2fg[:, t, :] + h2fg + c2fg).unsqueeze(-1)
      ig = torch.sigmoid(x2ig[:, t, :] + h2ig + c2ig).unsqueeze(-1)

      # Calculate memory cell blocks input activation and reshape to separate memory cell blocks.
      # Shape: (B, n_blk, d_blk).
      mc_in = torch.tanh(x2mc_in[:, t, :] + self.fc_h2mc_in(h_prev)).reshape(-1, self.n_blk, self.d_blk)

      # Calculate memory cell blocks' new internal states.
      # Shape: (B, n_blk, d_blk).
      c_cur = fg * c_prev + ig * mc_in

      # Calculate output gate units peephole connections.
      # shape: (B, n_blk).
      c2og = (self.pc_c2og * c_cur).sum(dim=-1)

      # Get output gate units and unsqueeze to separate memory cell blocks.
      # shape: (B, n_blk, 1).
      og = torch.sigmoid(x2og[:, t, :] + h2og + c2og).unsqueeze(-1)

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


class LSTM2002(LSTM2000):
  r"""LSTM (2002 version) language model.

  Implement RNN model in the paper `Learning Precise Timing with LSTM Recurrent Networks`_ as a language model.

  .. _`Learning Precise Timing with LSTM Recurrent Networks`: https://www.jmlr.org/papers/v3/gers02a.html

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

  LSTM (2002 version) language model is defined as follow:

  .. math::

    \newcommand{\br}[1]{\left[ #1 \right]}
    \newcommand{\eq}{\leftarrow}
    \newcommand{\pa}[1]{\left( #1 \right)}
    \newcommand{\cat}[1]{\operatorname{concate}\pa{#1}}
    \newcommand{\sof}[1]{\operatorname{softmax}\pa{#1}}
    \begin{align*}
      & \textbf{procedure }\text{LSTM2002}\pa{x, \br{h_0^0, c_0^0, \dots, h_0^{\nLyr-1}, c_0^{\nLyr-1}}}              \\
      & \hspace{1em} \textbf{for } t \in \set{0, \dots, S-1} \textbf{ do}                                             \\
      & \hspace{2em} e_t \eq (x_t)\text{-th row of } E \text{ but treated as column vector}                           \\
      & \hspace{2em} h_t^{-1} \eq \tanh(W_h \cdot e_t + b_h)                                                          \\
      & \hspace{1em} \textbf{end for}                                                                                 \\
      & \hspace{1em} h^{-1} \eq \cat{h_0^{-1}, \dots, h_{S-1}^{-1}}                                                   \\
      & \hspace{1em} \textbf{for } \ell \in \set{0, \dots, \nLyr-1} \textbf{ do}                                      \\
      & \hspace{2em} [h^\ell, c^\ell] \eq \text{LSTM2002Layer}(x \eq h^{\ell-1}, [h_0, c_0] \eq [h_0^\ell, c_0^\ell]) \\
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
  | :math:`\text{LSTM2002Layer}`                 | :math:`h_t^\ell` | :math:`(B, \dHid)`           |
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

  - The only differences between :py:class:`lmp.model.LSTM2000` and :py:class:`lmp.model.LSTM2002` are the underlying
    layers :py:class:`lmp.model.LSTM2000Layer` and :py:class:`lmp.model.LSTM2002Layer`.
    All other symbols are calculated as in :py:class:`lmp.model.LSTM2000`.

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
    CLI name of LSTM (2002 version) is ``LSTM-2002``.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.
  n_lyr: int
    Number of recurrent layers :math:`\nLyr`.
  p_emb: float
    Embeddings dropout probability.
  p_hid: float
    Hidden units dropout probability.
  stack_rnn: torch.nn.ModuleList
    :py:class:`lmp.model.LSTM2002Layer` stacking layers.
    Each LSTM (2002 version) layer is followed by a dropout layer with probability ``p_hid``.
    The number of stacking layers is equal to ``2 * n_lyr``.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dHid)`.

  See Also
  --------
  lmp.model.LSTM2000
    LSTM (2000 version) language model.
  lmp.model.LSTM2002Layer
    LSTM (2002 version) recurrent neural network.
  """

  model_name: ClassVar[str] = 'LSTM-2002'

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
    super().__init__(
      d_blk=d_blk,
      d_emb=d_emb,
      n_blk=n_blk,
      n_lyr=n_lyr,
      p_emb=p_emb,
      p_hid=p_hid,
      tknzr=tknzr,
      **kwargs,
    )

    # Stacking LSTM (2002 version) layers.
    # Each RNN layer is followed by one dropout layer.
    self.stack_rnn = nn.ModuleList([])
    for _ in range(n_lyr):
      self.stack_rnn.append(LSTM2002Layer(d_blk=d_blk, in_feat=self.d_hid, n_blk=n_blk))
      self.stack_rnn.append(nn.Dropout(p=p_hid))

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add LSTM (2002 version) language model hyperparameters to CLI arguments parser.

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
    >>> from lmp.model import LSTM2002
    >>> parser = argparse.ArgumentParser()
    >>> LSTM2002.add_CLI_args(parser)
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
    group = parser.add_argument_group('LSTM (2002 version) hyperparameters')
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
      help='Number of LSTM (2002 version) layers.',
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
