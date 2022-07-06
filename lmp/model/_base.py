"""Language model base class."""

import abc
import argparse
from typing import Any, ClassVar, List, Optional, Tuple

import torch


class BaseModel(abc.ABC, torch.nn.Module):
  r"""Language model abstract base class.

  Implement basic functionalities of language model, including training loss calculation, next token id prediction and
  parsing training arguments.

  Let :math:`X = \set{x^0, x^1, \dots, x^{B - 1}}` be a mini-batch of token id list with batch size :math:`B`.
  A token id list :math:`x \in X` is defined as follow:

  .. math::

     \newcommand{\pa}[1]{\left( #1 \right)}
     x = \pa{x[0], x[1], \dots, x[S]}.

  - :math:`x` has length :math:`S+1`.
  - :math:`x[t]` is the :math:`t`\-th time step of :math:`x`, the range of :math:`t` is :math:`\set{0, \dots, S}`.
  - :math:`x[0]` is the token id of ``[bos]``.
  - :math:`x[S]` is the token id of ``[eos]``.
  - Each language model will be paired with one tokenizer.
    Let :math:`V` be the size of the paired tokenizer's vocabulary.
    Then each :math:`x[t]` is assigned with an unique token in the tokenizer's vocabulary, i.e.,
    :math:`x[t] \in \set{0, \dots, V-1}`.

  The training goal of a language model with parameter :math:`\theta` is to find an optimal parameter
  :math:`\theta^{\star}`, such that when replace :math:`\theta` with :math:`\theta^{\star}`, it maximizes the
  prediction probability of next token id :math:`x[t]` given :math:`x[0], \dots, x[t-1]`:

  .. math::

     \theta^{\star} = \arg\max_{\theta} \prod_{x \in X} \prod_{t = 1}^S P(x[t] | x[0], \dots, x[t-1] ; \theta)

  Note that all token id lists in :math:`X` have the same length :math:`S+1`, and :math:`t` start with :math:`1`
  instead of :math:`0`.
  Thus for each token id list :math:`x`, the first :math:`S` token ids are served as input, and the last :math:`S`
  token ids are served as ground truth.
  There are only :math:`S` positions contribute to loss.

  Parameters
  ----------
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  model_name: typing.ClassVar[str]
    CLI name of the model.
    Only used to parse CLI arguments.
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
  def forward(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    r"""Calculate next token id logits.

    Logits were calculated based on previous hidden states and current input token id.
    Use :py:meth:`lmp.model.BaseModel.pred` to convert logits into next token id probability distribution over
    tokenizer's vocabulary.
    Use :py:meth:`lmp.model.BaseModel.loss` to convert logits into next token id prediction loss.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous calculation results.
      Set to ``None`` to use initial hidden states.
      Different models may have different hidden states structure.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id logits with shape :math:`(B, S, V)` and
      ``dtype == torch.float``.
      The second item in the tuple is a list of current hiddent states.
      Different models may have different hidden states structure.

    See Also
    --------
    lmp.tknzr.BaseTknzr.enc
      Source of token ids.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def loss(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Calculate language model prediction loss.

    Loss is defined as the **difference** between next token prediction and ground truth.
    This is treated as classification problem.
    The number of classes equals to the vocabulary size.
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
      Batch of previous calculation results.
      Set to ``None`` to use initial hidden states.
      Different models may have different hidden states structure.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the mini-batch loss with shape :math:`(1)` and ``dtype == torch.float``.
      The second item in the tuple is a list of current hiddent states.
      Different models may have different hidden states structure.
    """
    raise NotImplementedError

  @torch.no_grad()
  @abc.abstractmethod
  def pred(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Optional[List[torch.Tensor]] = None,
  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Calculate next token id probability distribution over tokenizer's vocabulary.

    Probabilities were calculated based on previous hidden states and current input token id.
    This method must only be used for inference.
    For training use :py:meth:`lmp.model.BaseModel.loss` instead.
    No tensor graphs will be constructed and no gradients will be calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Optional[list[torch.Tensor]], default: None
      Batch of previous calculation results.
      Set to ``None`` to use initial hidden states.
      Different models may have different hidden states structure.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
      The first item in the tuple is the batch of next token id probability distribution over the tokenizer's
      vocabulary.
      Probability tensor has shape :math:`(B, S, V)` and ``dtype == torch.float``.
      The second item in the tuple is a list of current hiddent states.
      Different models may have different hidden states structure.
    """
    raise NotImplementedError
