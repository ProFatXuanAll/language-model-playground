"""Language model base class."""

import abc
import argparse
from typing import Any, ClassVar, Tuple

import torch


class BaseModel(abc.ABC, torch.nn.Module):
  r"""Language model abstract base class.

  Implement basic functionalities of language model, including training loss calculation, next token id prediction and
  parsing training arguments.

  Let :math:`X = \set{x^1, x^2, \dots, x^B}` be a mini-batch of token id lists with batch size :math:`B`.
  A token id list :math:`x \in X` is defined as follow:

  .. math::

    x = \pa{x_1, x_2, \dots, x_S, x_{S+1}}.

  - :math:`x` has length :math:`S+1`.
  - :math:`x_t` is the :math:`t`\-th time step of :math:`x`, the range of :math:`t` is :math:`\set{1, \dots, S+1}`.
  - :math:`x_1` is the token id of ``<bos>``.
  - :math:`x_{S+1}` is the token id of ``<eos>``.
  - Each language model will be paired with one tokenizer.
    Let :math:`V` be the size of the paired tokenizer's vocabulary.
    Then :math:`x_t \in \set{1, \dots, V}`.

  The training goal of a language model with parameter :math:`\theta` is to find an optimal parameter
  :math:`\theta^{\star}`, such that when replace :math:`\theta` with :math:`\theta^{\star}`, it maximizes the
  prediction probability of next token id :math:`x_{t+1}` given :math:`x_1, \dots, x_t`:

  .. math::

     \theta^{\star} = \arg\max_{\theta} \prod_{x \in X} \prod_{t = 1}^S P(x_{t+1} \vert x_1, \dots, x_t ; \theta)

  Note that all token id lists in :math:`X` have the same length :math:`S+1` and :math:`t` start with :math:`1`.
  Thus for each token id list :math:`x \in X`, the first :math:`S` token ids are served as input, and the last
  :math:`S` token ids are served as prediction target.
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

  @classmethod
  @abc.abstractmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add language model hyperparameters to CLI argument parser.

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
    """
    raise NotImplementedError

  @abc.abstractmethod
  def cal_loss(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_next_tkids: torch.Tensor,
    batch_prev_states: Any = None,
  ) -> Tuple[torch.Tensor, Any]:
    """Calculate language model prediction loss.

    Loss is defined as the **next token id distribution difference** between model output and the answer.
    Predicting next token is treated as a classification problem, where the number of classes equals to tokenizer's
    vocabulary size.
    This method is only used for training.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_next_tkids: torch.Tensor
      Prediction target of each sample in the batch.
      ``batch_next_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Any, default: None
      Batch of previous calculation results.
      Set to ``None`` to use initial hidden states.
      Different models may have different hidden states structure.

    Returns
    -------
    tuple[torch.Tensor, typing.Any]
      The first item in the tuple is the mini-batch loss with shape :math:`(1)` and ``dtype == torch.float``.
      The second item in the tuple represent the current hidden states.
      Different models may have different hidden states structure.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def forward(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Any = None,
  ) -> Tuple[torch.Tensor, Any]:
    r"""Calculate next token id logits.

    Logits were calculated based on previous hidden states and current input token id.
    Use :py:meth:`~pred` to convert logits into next token id probability distribution over
    tokenizer's vocabulary.
    Use :py:meth:`~cal_loss` to convert logits into next token id prediction loss.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Any, default: None
      Batch of previous calculation results.
      Set to ``None`` to use initial hidden states.
      Different models may have different hidden states structure.

    Returns
    -------
    tuple[torch.Tensor, Any]
      The first item in the tuple is the batch of next token id logits with shape :math:`(B, S, V)` and
      ``dtype == torch.float``.
      The second item in the tuple represent the current hidden states.
      Different models may have different hidden states structure.

    See Also
    --------
    ~lmp.tknzr.BaseTknzr.enc
      Source of token ids.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def params_init(self) -> None:
    """Initialize model parameters.

    The ways and values to initialize a model are consider as hyperparameters.
    Different models may have different initialization sheme.

    Returns
    -------
    None
    """
    raise NotImplementedError

  @torch.no_grad()
  @abc.abstractmethod
  def pred(
    self,
    batch_cur_tkids: torch.Tensor,
    batch_prev_states: Any = None,
  ) -> Tuple[torch.Tensor, Any]:
    """Calculate next token id probability distribution over tokenizer's vocabulary.

    Probability distribution is calculated based on previous hidden states and current input token id.
    This method is only used for inference.
    For training use :py:meth:`~cal_loss` instead.
    No tensor graphs are constructed and no gradients are calculated.

    Parameters
    ----------
    batch_cur_tkids: torch.Tensor
      Batch of current input token ids.
      ``batch_cur_tkids`` has shape :math:`(B, S)` and ``dtype == torch.long``.
    batch_prev_states: typing.Any, default: None
      Batch of previous calculation results.
      Set to ``None`` to use initial hidden states.
      Different models may have different hidden states structure.

    Returns
    -------
    tuple[torch.Tensor, Any]
      The first item in the tuple is the batch of next token id probability distributions over the tokenizer's
      vocabulary.
      Probability tensor has shape :math:`(B, S, V)` and ``dtype == torch.float``.
      The second item in the tuple represent the current hidden states.
      Different models may have different hidden states structure.
    """
    raise NotImplementedError
