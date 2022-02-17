"""Language model base class."""

import abc
import argparse
from typing import Any, ClassVar, List, Optional, Tuple

import torch


class BaseModel(abc.ABC, torch.nn.Module):
  r"""Language model abstract base class.

  Implement basic functionalities for language model, including training loss calculation, next token id prediction
  and parsing training arguments.

  We define the input token id list :math:`x` as follow:

  .. math::

     \newcommand{\pa}[1]{\left( #1 \right)}
     \newcommand{\set}[1]{\left\lbrace #1 \right\rbrace}
     x = \pa{x[1], x[2], \dots, x[S]}.

  - :math:`x` has length :math:`S`.
  - :math:`x[1]` is the token id of ``[bos]``.
  - :math:`x[S]` is the token id of ``[eos]``.

  Let :math:`x[t]` be the :math:`t`-th time step of :math:`x` where :math:`t \in \set{1, \dots, S}`.  The
  training goal of a language model with parameter :math:`\theta` is to find an optimal parameter
  :math:`\theta^{\star}` such that when replace  :math:`\theta` with :math:`\theta^{\star}` it maximizes the prediction
  probability of next token id :math:`x[t]` given :math:`x[1], \dots, x[t - 1]`:

  .. math::

     \theta^{\star} = \arg\max_{\theta} \prod_{t = 1}^S P(x[t] | x[1], \dots, x[t - 1] ; \theta)

  Parameters
  ----------
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.

  Attributes
  ----------
  model_name: typing.ClassVar[str]
    CLI Display name of the model.  Only used to parse CLI arguments.
  """

  model_name: ClassVar[str] = 'base'

  def __init__(self, **kwargs: Any):
    super().__init__()

  @abc.abstractmethod
  def params_init(self) -> None:
    """Initialize model parameters.

    Returns
    -------
    None
    """
    raise NotImplementedError

  @classmethod
  @abc.abstractmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add language model constructor parameters to CLI arguments parser.

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
    """
    raise NotImplementedError

  @abc.abstractmethod
  def forward(self, batch_cur_tkids: torch.Tensor, batch_next_tkids: torch.Tensor) -> torch.Tensor:
    """Calculate language model training loss.

    This method must only be used to train model.  For inference use :py:meth:`lmp.model.BaseModel.pred` instead.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of token ids which represent input token ids of all time steps.  ``batch_cur_tkids`` has shape
      ``(batch_size, seq_len)`` and ``dtype == torch.long``.
    batch_next_tkids: torch.Tensor
      Batch of token ids which represent prediction targets of all time steps.  ``batch_next_tkids`` has shape
      ``(batch_size, seq_len)`` and ``dtype == torch.long``.

    Returns
    -------
    torch.Tensor
      Mini-batch loss of next token id prediction.  Returned tensor has shape ``(1)`` and ``dtype == torch.float``.

    See Also
    --------
    lmp.tknzr.BaseTknzr.enc
      Token encoding was done by tokenizers.
    """
    raise NotImplementedError

  @torch.no_grad()
  @abc.abstractmethod
  def pred(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Calculate next token id probability distribution given previous hidden states and current input token id.

    This method must only be used for inference.  For training use :py:meth:`lmp.model.BaseModel.forward` instead.  No
    tensor graphs will be constructed and no gradients will be calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.  ``batch_cur_tkids`` has shape ``(batch_size)`` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous calculation results.  Set to ``None`` to use initial hidden states.  Different models have
      different hidden states structure.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the tensor of batch of next token id probability distribution with shape
      ``(batch_size, vocab_size)`` and ``dtype == torch.float``.  The second item in the tuple is a list of tensor
      which represent current hiddent states.  Different models have different hidden states structure.
    """
    raise NotImplementedError
