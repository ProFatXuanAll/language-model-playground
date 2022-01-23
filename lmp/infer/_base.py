"""Inference method base class."""

import abc
import argparse
from typing import Any, ClassVar

import torch

from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


class BaseInfer(abc.ABC):
  """Inference method abstract base class.

  All inference methods must inherit :py:class:`lmp.infer.BaseInfer`.

  For comments throughout this class and its subclasses, we use the following
  symbols to denote the shape of tensors:

  Parameters
  ----------
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.
  max_seq_len: str
    Generated sequence of tokens maximum sequence length constraint.  Must satisfy
    ``-1 <= max_seq_len <= BaseInfer.hard_max_seq_len``.  If ``max_seq_len == -1``, then replace ``max_seq_len`` with
    ``BaseInfer.hard_max_seq_len``.  Raise ``ValueError`` if constraint is violated.

  Attributes
  ----------
  hard_max_seq_len: ClassVar[int]
    Hard limit of maximum sequence length.  This is set to avoid generating too many tokens.
  infer_name: ClassVar[str]
    Display name for inference method on CLI.  Used for command line argument parsing.
  max_seq_len: str
    Maximum sequence length constraint of generated sequence of tokens.

  Raises
  ------
  TypeError
    If ``max_seq_len`` is not an instance of :py:class:`int`.
  ValueError
    If ``max_seq_len`` is not in ``range(-1, BaseInfer.hard_max_seq_len)``.
  """

  hard_max_seq_len: ClassVar[int] = 512
  infer_name: ClassVar[str] = 'base'

  def __init__(self, max_seq_len: int, **kwargs: Any):
    if not isinstance(max_seq_len, int):
      raise TypeError('`max_seq_len` must be an instance of `int`.')

    # `max_seq_len` must be valid.
    if max_seq_len == -1:
      self.max_seq_len = self.__class__.hard_max_seq_len
    elif not (0 <= max_seq_len <= self.__class__.hard_max_seq_len):
      raise ValueError('`max_seq_len` must be in the range from 0 to ' + f'{self.__class__.hard_max_seq_len}.')
    else:
      self.max_seq_len = max_seq_len

  @torch.no_grad()
  @abc.abstractmethod
  def gen(self, model: BaseModel, tknzr: BaseTknzr, txt: str) -> str:
    """Generate text conditional on text segment.

    Parameters
    ----------
    model: lmp.model.BaseModel
      Pre-trained language model to generate text.
    tknzr: lmp.tknzr.BaseTknzr
      Pre-trained tokenizer for text segment encoding.
    txt: str
      Text segment to condition on.

    Returns
    -------
    str
      Generated text.
    """
    raise NotImplementedError

  @staticmethod
  def infer_parser(parser: argparse.ArgumentParser) -> None:
    """Language model text generation CLI arguments parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      Parser for CLI arguments.

    See Also
    --------
    lmp.script.generate_text
      Generate text using pre-trained language model.

    Examples
    --------
    >>> import argparse
    >>> from lmp.infer import BaseInfer
    >>> parser = argparse.ArgumentParser()
    >>> BaseInfer.infer_parser(parser)
    >>> args = parser.parse_args([
    ...   '--ckpt', '5000',
    ...   '--exp_name', 'my_exp',
    ...   '--txt', 'Hello world',
    ... ])
    >>> args.ckpt == 5000
    True
    >>> args.exp_name == 'my_exp'
    True
    >>> args.txt == 'Hello world'
    True
    >>> args.seed == 42
    True
    """
    # Required arguments.
    group = parser.add_argument_group('common arguments')
    group.add_argument(
      '--ckpt',
      help='Pre-trained language model checkpoint to inference.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--exp_name',
      help='Pre-trained language model experiment name.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--txt',
      help='Text segment to conditional on.',
      required=True,
      type=str,
    )

    # Optional arguments.
    group.add_argument(
      '--seed',
      default=42,
      help='Random seed.',
      type=int,
    )
