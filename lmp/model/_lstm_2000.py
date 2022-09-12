"""LSTM (2000 version) language model."""

import argparse
from typing import Any, ClassVar, Optional, Tuple

import torch
import torch.nn as nn

import lmp.util.metric
import lmp.util.validate
from lmp.model._lstm_1997 import LSTM1997, LSTM1997Layer
from lmp.tknzr._base import BaseTknzr


class LSTM2000(LSTM1997):
  r"""LSTM (2000 version) :footcite:`gers2000lstm` language model.

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

  LSTM (2000 version) language model is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\LSTMZeroZero}\pa{x, \pa{\br{c_0^1, \dots, c_0^{\nLyr}}, \br{h_0^1, \dots, h_0^{\nLyr}}}} \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                                        \\
      & \indent{2} e_t \algoEq (x_t)\text{-th row of } E \text{ but treated as column vector}               \\
      & \indent{2} \widehat{e_t} \algoEq \drop{e_t}{\pEmb}                                                  \\
      & \indent{2} h_t^0 \algoEq \tanh\pa{W_h \cdot \widehat{e_t} + b_h}                                    \\
      & \indent{1} \algoEndFor                                                                              \\
      & \indent{1} h^0 \algoEq \cat{h_1^0, \dots, h_S^0}                                                    \\
      & \indent{1} \widehat{h^0} \algoEq \drop{h^0}{\pHid}                                                  \\
      & \indent{1} \algoFor{\ell \in \set{1, \dots, \nLyr}}                                                 \\
      & \indent{2} \pa{c^\ell, h^\ell} \algoEq \LSTMZeroZeroLayer\pa{
                                                 x \algoEq \widehat{h^{\ell-1}},
                                                 c_0 \algoEq c_0^\ell,
                                                 h_0 \algoEq h_0^\ell
                                               }                                                            \\
      & \indent{2} \widehat{h^\ell} \algoEq \drop{h^\ell}{\pHid}                                            \\
      & \indent{1} \algoEndFor                                                                              \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                                        \\
      & \indent{2} z_t \algoEq \tanh\pa{W_z \cdot h_t^{\nLyr} + b_z}                                        \\
      & \indent{2} \widehat{z_t} \algoEq \drop{z_t}{\pHid}                                                  \\
      & \indent{2} y_t \algoEq \sof{E \cdot \widehat{z_t}}                                                  \\
      & \indent{1} \algoEndFor                                                                              \\
      & \indent{1} y \algoEq \cat{y_1, \dots, y_S}                                                          \\
      & \indent{1} \algoReturn \pa{y, \pa{\br{c_S^1, \dots, c_S^{\nLyr}}, \br{h_S^1, \dots, h_S^{\nLyr}}}}  \\
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
  | :math:`\LSTMZeroZeroLayer`                | :math:`\widehat{h^\ell}` | :math:`(B, \dHid)`           |
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

  - The only differences between :py:class:`~lmp.model.LSTM1997` and :py:class:`~LSTM2000` are the underlying layers
    :py:class:`~lmp.model.LSTM1997Layer` and :py:class:`~LSTM2000Layer`.
    All other symbols are calculated as in :py:class:`lmp.model.LSTM1997`.
  - Forget gate biases are initialized with uniform distribution :math:`\mathcal{U}(0, \init_{fb})`.
    This make forget gate remain open at the start of training.

  Parameters
  ----------
  d_blk: int, default: 1
    Number of units in a memory cell block :math:`\dBlk`.
  d_emb: int, default: 1
    Token embedding dimension :math:`\dEmb`.
  init_ib: float, default: 1.0
    Uniform distribution upper bound :math:`\init_{fb}` used to initialize forget gate biases.
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
  tknzr: ~lmp.tknzr.BaseTknzr
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
  init_fb: float
    Uniform distribution upper bound :math:`\init_{fb}` used to initialize forget gate biases.
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
    CLI name of LSTM (2000 version) is ``LSTM-2000``.
  n_blk: int
    Number of memory cell blocks :math:`\nBlk`.
  n_lyr: int
    Number of recurrent layers :math:`\nLyr`.
  p_emb: float
    Embeddings dropout probability :math:`\pEmb`.
  p_hid: float
    Hidden units dropout probability :math:`\pHid`.
  stack_rnn: torch.nn.ModuleList
    :py:class:`~LSTM2000Layer` stacking layers.
    Each LSTM (2000 version) layer is followed by a dropout layer with probability :math:`\pHid`.
    The number of stacking layers is equal to :math:`2 \nLyr`.
    Input shape: :math:`(B, S, \dHid)`.
    Output shape: :math:`(B, S, \dHid)`.

  See Also
  --------
  ~lmp.model.LSTM1997
    LSTM (1997 version) language model.
  ~lmp.model.LSTM1997Layer
    LSTM (1997 version) recurrent neural network.
  ~LSTM2000Layer
    LSTM (2000 version) recurrent neural network.
  """

  model_name: ClassVar[str] = 'LSTM-2000'

  def __init__(
    self,
    *,
    d_blk: int = 1,
    d_emb: int = 1,
    init_fb: float = 1.0,
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
    super().__init__(
      d_blk=d_blk,
      d_emb=d_emb,
      init_ib=init_ib,
      init_lower=init_lower,
      init_ob=init_ob,
      init_upper=init_upper,
      label_smoothing=label_smoothing,
      n_blk=n_blk,
      n_lyr=n_lyr,
      p_emb=p_emb,
      p_hid=p_hid,
      tknzr=tknzr,
      **kwargs,
    )

    # `init_fb` validation.
    lmp.util.validate.raise_if_not_instance(val=init_fb, val_name='init_fb', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0, init_fb], val_names=['0', 'init_fb'])
    self.init_fb = init_fb

    # Stacking LSTM (2000 version) layers.
    # Each RNN layer is followed by one dropout layer.
    self.stack_rnn = nn.ModuleList([])
    for _ in range(n_lyr):
      self.stack_rnn.append(
        LSTM2000Layer(
          d_blk=d_blk,
          in_feat=self.d_hid,
          init_fb=init_fb,
          init_lower=init_lower,
          init_ob=init_ob,
          init_upper=init_upper,
          n_blk=n_blk,
        )
      )
      self.stack_rnn.append(nn.Dropout(p=p_hid))

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add LSTM (2000 version) language model hyperparameters to CLI argument parser.

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
    >>> from lmp.model import LSTM2000
    >>> parser = argparse.ArgumentParser()
    >>> LSTM2000.add_CLI_args(parser)
    >>> args = parser.parse_args([
    ...   '--d_blk', '64',
    ...   '--d_emb', '100',
    ...   '--init_fb', '0.1',
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
    >>> assert math.isclose(args.init_fb, 0.1)
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
    group = parser.add_argument_group('LSTM (2000 version) language model hyperparameters')
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
      '--init_fb',
      default=1.0,
      help='''
      Uniform distribution upper bound used to initialize forget gate biases.
      Default is ``1.0``.
      ''',
      type=float,
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
      Number of LSTM (2000 version) layers.
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


class LSTM2000Layer(LSTM1997Layer):
  r"""LSTM (2000 version) :footcite:`gers2000lstm` recurrent neural network.

  - Let :math:`\hIn` be the number of input features per time step.
  - Let :math:`\dBlk` be the number of units in a memory cell block.
  - Let :math:`\nBlk` be the number of memory cell blocks.
  - Let :math:`\hOut = \nBlk \times \dBlk` be the number of output features per time step.
  - Let :math:`x` be a batch of sequences of input features with shape :math:`(B, S, \hIn)`, where :math:`B` is batch
    size and :math:`S` is per sequence length.
  - Let :math:`h_0` be the initial hidden states with shape :math:`(B, \hOut)`.
  - Let :math:`c_0` be the memory cell initial internal states with shape :math:`(B, \nBlk, \dBlk)`.

  LSTM (2000 version) layer is defined as follow:

  .. math::

    \begin{align*}
      & \algoProc{\LSTMZeroZeroLayer}\pa{x, c_0, h_0}                                                   \\
      & \indent{1} S \algoEq x.\sz{1}                                                                   \\
      & \indent{1} \algoFor{t \in \set{1, \dots, S}}                                                    \\
      & \indent{2} f_t \algoEq \sigma(W_f \cdot x_t + U_f \cdot h_{t-1} + b_f)       &&\tag{1}\label{1} \\
      & \indent{2} i_t \algoEq \sigma(W_i \cdot x_t + U_i \cdot h_{t-1} + b_i)                          \\
      & \indent{2} o_t \algoEq \sigma(W_o \cdot x_t + U_o \cdot h_{t-1} + b_o)                          \\
      & \indent{2} \algoFor{k \in \set{1, \dots, \nBlk}}                                                \\
      & \indent{3} g_{t,k} \algoEq \tanh\pa{W_k \cdot x_t + U_k \cdot h_{t-1} + b_k}                    \\
      & \indent{3} c_{t,k} \algoEq f_{t, k} \cdot c_{t-1,k} + i_{t,k} \cdot g_{t,k}  &&\tag{2}\label{2} \\
      & \indent{3} h_{t,k} \algoEq o_{t,k} \cdot \tanh\pa{c_{t,k}}                                      \\
      & \indent{2} \algoEndFor                                                                          \\
      & \indent{2} c_t \algoEq \cat{c_{t,1}, \dots, c_{t,\nBlk}}                                        \\
      & \indent{2} h_t \algoEq \fla{h_{t,1}, \dots, h_{t,\nBlk}}                                        \\
      & \indent{1} \algoEndFor                                                                          \\
      & \indent{1} c \algoEq \cat{c_1, \dots, c_S}                                                      \\
      & \indent{1} h \algoEq \cat{h_1, \dots, h_S}                                                      \\
      & \indent{1} \algoReturn (c, h)                                                                   \\
      & \algoEndProc
    \end{align*}

  +--------------------------------------+------------------------------------------------+
  | Trainable Parameters                 | Nodes                                          |
  +-------------+------------------------+-----------------+------------------------------+
  | Parameter   | Shape                  | Symbol          | Shape                        |
  +=============+========================+=================+==============================+
  | :math:`U_f` | :math:`(\nBlk, \hOut)` | :math:`c`       | :math:`(B, S, \nBlk, \dBlk)` |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`U_i` | :math:`(\nBlk, \hOut)` | :math:`c_t`     | :math:`(B, \nBlk, \dBlk)`    |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`U_k` | :math:`(\dBlk, \hOut)` | :math:`c_{t,k}` | :math:`(B, \dBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`U_o` | :math:`(\nBlk, \hOut)` | :math:`f_t`     | :math:`(B, \nBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`W_f` | :math:`(\nBlk, \hIn)`  | :math:`f_{t,k}` | :math:`(B, 1)`               |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`W_i` | :math:`(\nBlk, \hIn)`  | :math:`g_{t,k}` | :math:`(B, \dBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`W_k` | :math:`(\dBlk, \hIn)`  | :math:`h`       | :math:`(B, S, \hOut)`        |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`W_o` | :math:`(\nBlk, \hIn)`  | :math:`h_t`     | :math:`(B, \hOut)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`b_f` | :math:`(\nBlk)`        | :math:`h_{t,k}` | :math:`(B, \dBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`b_i` | :math:`(\nBlk)`        | :math:`i_t`     | :math:`(B, \nBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`b_k` | :math:`(\dBlk)`        | :math:`i_{t,k}` | :math:`(B, 1)`               |
  +-------------+------------------------+-----------------+------------------------------+
  | :math:`b_o` | :math:`(\nBlk)`        | :math:`o_t`     | :math:`(B, \nBlk)`           |
  +-------------+------------------------+-----------------+------------------------------+
  |                                      | :math:`o_{t,k}` | :math:`(B, 1)`               |
  |                                      +-----------------+------------------------------+
  |                                      | :math:`x`       | :math:`(B, S, \hIn)`         |
  |                                      +-----------------+------------------------------+
  |                                      | :math:`x_t`     | :math:`(B, \hIn)`            |
  +--------------------------------------+-----------------+------------------------------+

  - The only differences between :py:class:`~lmp.model.LSTM1997Layer` and :py:class:`~LSTM2000Layer` are equations
    :math:`\eqref{1}\eqref{2}`.
    All other symbols are calculated as in :py:class:`~lmp.model.LSTM1997Layer`.
  - :math:`f_t` is memory cell blocks' forget gate units at time step :math:`t`.
    :math:`f_{t,k}` is the :math:`k`-th coordinates of :math:`f_t`, which represents the :math:`k`-th memory cell
    block's forget gate unit at time step :math:`t`.
  - Model parameters in LSTM (2000 version) layer are initialized as in :py:class:`~lmp.model.LSTM1997Layer`.
  - Forget gate biases are initialized with uniform distribution :math:`\mathcal{U}(0, \init_{fb})`.
    The upper bound :math:`\init_{fb}` is given as hyperparameter.
    This make forget gate remain open at the start of training.

  Parameters
  ----------
  d_blk: int, default: 1
    Dimension of each memory cell block :math:`\dBlk`.
  in_feat: int, default: 1
    Number of input features per time step :math:`\hIn`.
  init_fb: float, default: 1.0
    Uniform distribution upper bound :math:`\init_{fb}` used to initialize forget gate biases.
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
  fc_h2fg: torch.nn.Linear
    Fully connected layer :math:`U_f` which connects hidden states to memory cell's forget gate units.
    Input shape: :math:`(B, \dHid)`.
    Output shape: :math:`(B, \nBlk)`.
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
  fc_x2fg: torch.nn.Linear
    Fully connected layer :math:`W_f` and :math:`b_f` which connects input units to memory cell's forget gate units.
    Input shape: :math:`(B, S, \hIn)`.
    Output shape: :math:`(B, S, \nBlk)`.
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
  init_fb: float
    Uniform distribution upper bound :math:`\init_{fb}` used to initialize forget gate biases.
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

  See Also
  --------
  ~lmp.model.LSTM1997Layer
    LSTM (1997 version) recurrent neural network.
  """

  def __init__(
    self,
    *,
    d_blk: int = 1,
    in_feat: int = 1,
    init_fb: float = 1.0,
    init_ib: float = -1.0,
    init_lower: float = -0.1,
    init_ob: float = -1.0,
    init_upper: float = 0.1,
    n_blk: int = 1,
    **kwargs: Any,
  ):
    super().__init__(
      d_blk=d_blk,
      in_feat=in_feat,
      init_ib=init_ib,
      init_lower=init_lower,
      init_ob=init_ob,
      init_upper=init_upper,
      n_blk=n_blk,
      **kwargs,
    )

    # `init_fb` validation.
    lmp.util.validate.raise_if_not_instance(val=init_fb, val_name='init_fb', val_type=float)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0, init_fb], val_names=['0', 'init_fb'])
    self.init_fb = init_fb

    # Fully connected layer which connects input units to forget gate units.
    self.fc_x2fg = nn.Linear(in_features=in_feat, out_features=n_blk)

    # Fully connected layer which connects hidden states to forget gate units.
    # Set `bias=False` to share bias term with `self.fc_x2fg` layer.
    self.fc_h2fg = nn.Linear(in_features=self.d_hid, out_features=n_blk, bias=False)

  def forward(
    self,
    x: torch.Tensor,
    c_0: Optional[torch.Tensor] = None,
    h_0: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculate batch of hidden states for ``x``.

    Below we describe the forward pass algorithm of LSTM (2000 version) layer.

    #. Let ``x`` be a batch of sequences of input features :math:`x`.
    #. Let ``x.size(1)`` be sequence length :math:`S`.
    #. Let ``c_0`` be the memory cell initial internal states :math:`c_0`.
       If ``c_0 is None``, use ``self.c_0`` instead.
    #. Let ``h_0`` be the initial hidden states :math:`h_0`.
       If ``h_0 is None``, use ``self.h_0`` instead.
    #. Loop through :math:`\set{1, \dots, S}` with looping index :math:`t`.

       #. Use :math:`x_t`, :math:`h_{t-1}`, ``self.fc_x2fg`` and ``self.fc_h2fg`` to get forget gate units :math:`f_t`.
       #. Use :math:`x_t`, :math:`h_{t-1}`, ``self.fc_x2ig`` and ``self.fc_h2ig`` to get input gate units :math:`i_t`.
       #. Use :math:`x_t`, :math:`h_{t-1}`, ``self.fc_x2og`` and ``self.fc_h2og`` to get output gate units :math:`o_t`.
       #. Use :math:`x_t`, :math:`h_{t-1}`, ``self.fc_x2mc_in`` and ``self.fc_h2mc_in`` to get memory cell input
          activations :math:`g_{t,1}, \dots, g_{t,\nBlk}`.
       #. Derive memory cell new internal state :math:`c_{t,1}, \dots, c_{t,\nBlk}` using forget gates units
          :math:`f_{t,1}, \dots, f_{t,\nBlk}`, input gate units :math:`i_{t,1}, \dots, i_{t,\nBlk}`, memory cell old
          internal states :math:`c_{t-1,1}, \dots, c_{t-1,\nBlk}` and memory cell input activations
          :math:`g_{t,1}, \dots, g_{t,\nBlk}`.
       #. Derive new hidden states :math:`h_t` using output gate units :math:`o_{t,1}, \dots, o_{t,\nBlk}` and memory
          cell new internal states :math:`c_{t,1}, \dots, c_{t,\nBlk}`.

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
    x2fg = self.fc_x2fg(x)
    x2ig = self.fc_x2ig(x)
    x2og = self.fc_x2og(x)

    # Transform input features to memory cell block's input.
    # Shape: (B, S, d_hid).
    x2mc_in = self.fc_x2mc_in(x)

    # Perform recurrent calculation for `S` steps.
    c_all = []
    h_all = []
    for t in range(S):
      # Get forget / input / output gate units and unsqueeze to separate memory cell blocks.
      # Shape: (B, n_blk, 1).
      fg = torch.sigmoid(x2fg[:, t, :] + self.fc_h2fg(h_prev)).unsqueeze(-1)
      ig = torch.sigmoid(x2ig[:, t, :] + self.fc_h2ig(h_prev)).unsqueeze(-1)
      og = torch.sigmoid(x2og[:, t, :] + self.fc_h2og(h_prev)).unsqueeze(-1)

      # Calculate memory cell blocks input activation and reshape to separate memory cell blocks.
      # Shape: (B, n_blk, d_blk).
      mc_in = torch.tanh(x2mc_in[:, t, :] + self.fc_h2mc_in(h_prev)).reshape(-1, self.n_blk, self.d_blk)

      # Calculate memory cell new internal states.
      # Shape: (B, n_blk, d_blk).
      c_cur = fg * c_prev + ig * mc_in

      # Calculate memory cell outputs and flatten to form the new hidden states.
      # Shape: (B, d_hid).
      h_cur = (og * torch.tanh(c_cur)).reshape(-1, self.d_hid)

      c_all.append(c_cur)
      h_all.append(h_cur)

      # Update hidden states and memory cell internal states.
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

    return (c, h)

  def params_init(self) -> None:
    r"""Initialize model parameters.

    All weights and biases other than :math:`b_f, b_i, b_o` are initialized with uniform distribution
    :math:`\mathcal{U}\pa{\init_l, \init_u}`.
    :math:`b_f` is initialized with uniform distribution :math:`\mathcal{U}\pa{0, \init_{fb}}`.
    :math:`b_i` is initialized with uniform distribution :math:`\mathcal{U}\pa{\init_{ib}, 0}`.
    :math:`b_o` is initialized with uniform distribution :math:`\mathcal{U}\pa{\init_{ob}, 0}`.
    :math:`b_f` is initialized separatedly so that forget gates remain open at the start of training.
    :math:`b_i, b_o` are initialized separatedly so that input and output gates remain closed at the start of training.

    Returns
    -------
    None

    See Also
    --------
    ~lmp.model.LSTM1997Layer.params_init
      LSTM (1997 version) layer parameter initialization.
    """
    super().params_init()

    # Initialize weights and biases with uniform distribution.
    nn.init.uniform_(self.fc_x2fg.weight, self.init_lower, self.init_upper)
    nn.init.uniform_(self.fc_h2fg.weight, self.init_lower, self.init_upper)

    # Forget gate units' biases are initialized to positive values.
    nn.init.uniform_(self.fc_x2fg.bias, 0.0, self.init_fb)
