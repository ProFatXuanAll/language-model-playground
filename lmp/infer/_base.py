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
    Maximum length constraint on generated token list.
    One can use larger contraint compare to training.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  infer_name: ClassVar[str]
    CLI name of the inference method.
    Only used to parse CLI arguments.
  max_seq_len: str
    Maximum length constraint of generated token list.

  See Also
  --------
  :doc:`lmp.infer </infer/index>`
    All available inference methods.
  """

  infer_name: ClassVar[str] = 'base'

  def __init__(self, max_seq_len: int, **kwargs: Any):
    # `max_seq_len` validation.
    lmp.util.validate.raise_if_not_instance(val=max_seq_len, val_name='max_seq_len', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, max_seq_len, 1024], val_names=['1', 'max_seq_len', '1024'])

    self.max_seq_len = max_seq_len

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add inference method constructor parameters to CLI arguments parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI arguments parser.

    Returns
    -------
    None

    See Also
    --------
    :doc:`lmp.script.gen_txt </script/gen_txt>`
      Use pre-trained language model checkpoint to generate continual text of given text segment.
    """
    # `parser` validation.
    lmp.util.validate.raise_if_not_instance(val=parser, val_name='parser', val_type=argparse.ArgumentParser)

  @torch.no_grad()
  @abc.abstractmethod
  def gen(self, model: BaseModel, tknzr: BaseTknzr, txt: str) -> str:
    """Generate continual text conditioned on given text segment.

    Parameters
    ----------
    model: lmp.model.BaseModel
      Pre-trained language model which will be used to generate text.
    tknzr: lmp.tknzr.BaseTknzr
      Pre-trained tokenizer which performs text encoding and decoding.
    txt: str
      Text segment which the generation process is conditioned on.

    Returns
    -------
    str
      Generated text.
    """
    raise NotImplementedError
