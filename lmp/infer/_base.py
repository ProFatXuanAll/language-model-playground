"""Inference method base class."""

import abc
import argparse
from typing import Any, ClassVar

import torch

import lmp.util.validate
from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


class BaseInfer(abc.ABC):
  """Inference method abstract base class.

  Implement basic functionalities for language model inference, including text generation and parsing inference
  arguments.

  Parameters
  ----------
  max_seq_len: str
    Maximum length constraint on generated token list.  One can use larger contraint compare to training.
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.

  Attributes
  ----------
  infer_name: ClassVar[str]
    CLI Display name of the inference method.  Only used to parse CLI arguments.
  max_seq_len: str
    Maximum length constraint of generated token list.

  See Also
  --------
  lmp.infer
    All available inference methods.
  """

  infer_name: ClassVar[str] = 'base'

  def __init__(self, max_seq_len: int, **kwargs: Any):
    # `max_seq_len` validation.
    lmp.util.validate.raise_if_not_instance(val=max_seq_len, val_name='max_seq_len', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, max_seq_len, 1024], val_names=['1', 'max_seq_len', '1024'])

    self.max_seq_len = max_seq_len

  @torch.no_grad()
  @abc.abstractmethod
  def gen(self, model: BaseModel, tknzr: BaseTknzr, txt: str) -> str:
    """Generate continual text conditioned on given text segment.

    Parameters
    ----------
    model: lmp.model.BaseModel
      Pre-trained language model which will be used to generate text.
    tknzr: lmp.tknzr.BaseTknzr
      Pre-trained tokenizer which perform text encoding and decoding.
    txt: str
      Text segment which the generation process is conditioned on.

    Returns
    -------
    str
      Generated text.
    """
    raise NotImplementedError

  @classmethod
  def infer_parser(cls, parser: argparse.ArgumentParser) -> None:
    """CLI arguments parser for language model text generation.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI arguments parser.

    Returns
    -------
    None

    See Also
    --------
    lmp.script.gen_txt
      Use pre-trained language model checkpoint to generate continual text of given text segment.

    Examples
    --------
    >>> import argparse
    >>> from lmp.infer import BaseInfer
    >>> parser = argparse.ArgumentParser()
    >>> BaseInfer.infer_parser(parser)
    >>> args = parser.parse_args([
    ...   '--ckpt', '5000',
    ...   '--exp_name', 'my_exp',
    ...   '--max_seq_len', '512',
    ...   '--txt', 'Hello world',
    ... ])
    >>> args.ckpt == 5000
    True
    >>> args.exp_name == 'my_exp'
    True
    >>> args.max_seq_len == 512
    True
    >>> args.txt == 'Hello world'
    True
    >>> args.seed == 42
    True
    """
    # Required arguments.
    group = parser.add_argument_group('language model inference arguments')
    group.add_argument(
      '--ckpt',
      help='Pre-trained language model checkpoint.',
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
      '--max_seq_len',
      help='Maximum sequence length constraint.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--txt',
      help='Text segment which the generation process is condition on.',
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
