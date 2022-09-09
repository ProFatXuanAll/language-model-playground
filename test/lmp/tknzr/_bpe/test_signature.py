"""Test :py:mod:`lmp.tknzr._bpe` signatures."""

import inspect
import re
from inspect import Parameter, Signature
from typing import Any, ClassVar, get_type_hints

import lmp.tknzr._bpe
from lmp.tknzr._base import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.tknzr._bpe, 'BPETknzr')
  assert inspect.isclass(lmp.tknzr._bpe.BPETknzr)
  assert not inspect.isabstract(lmp.tknzr._bpe.BPETknzr)
  assert issubclass(lmp.tknzr._bpe.BPETknzr, BaseTknzr)

  assert hasattr(lmp.tknzr._bpe, 'EOW_TK')
  assert isinstance(lmp.tknzr._bpe.EOW_TK, str)
  assert lmp.tknzr._bpe.EOW_TK == '<eow>'

  assert hasattr(lmp.tknzr._bpe, 'SPLIT_PTTN')
  assert isinstance(lmp.tknzr._bpe.SPLIT_PTTN, re.Pattern)
  assert lmp.tknzr._bpe.SPLIT_PTTN.pattern == r'(<bos>|<eos>|<pad>|<unk>|\s+)'


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(lmp.tknzr._bpe.BPETknzr) == {'tknzr_name': ClassVar[str]}
  assert lmp.tknzr._bpe.BPETknzr.tknzr_name == 'BPE'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert inspect.signature(lmp.tknzr._bpe.BPETknzr.add_CLI_args) == inspect.signature(BaseTknzr.add_CLI_args)


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are same as base class."""
  assert lmp.tknzr._bpe.BPETknzr.dec == BaseTknzr.dec
  assert lmp.tknzr._bpe.BPETknzr.enc == BaseTknzr.enc
  assert lmp.tknzr._bpe.BPETknzr.norm == BaseTknzr.norm
  assert lmp.tknzr._bpe.BPETknzr.pad_to_max == BaseTknzr.pad_to_max
  assert lmp.tknzr._bpe.BPETknzr.vocab_size == BaseTknzr.vocab_size


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert inspect.signature(lmp.tknzr._bpe.BPETknzr.__init__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=bool,
        default=False,
        kind=Parameter.KEYWORD_ONLY,
        name='is_uncased',
      ),
      Parameter(
        annotation=int,
        default=-1,
        kind=Parameter.KEYWORD_ONLY,
        name='max_vocab',
      ),
      Parameter(
        annotation=int,
        default=0,
        kind=Parameter.KEYWORD_ONLY,
        name='min_count',
      ),
      Parameter(
        annotation=int,
        default=10000,
        kind=Parameter.KEYWORD_ONLY,
        name='n_merge',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert inspect.signature(lmp.tknzr._bpe.BPETknzr.build_vocab) == inspect.signature(BaseTknzr.build_vocab)
  assert lmp.tknzr._bpe.BPETknzr.build_vocab != BaseTknzr.build_vocab

  assert inspect.signature(lmp.tknzr._bpe.BPETknzr.dtknz) == inspect.signature(BaseTknzr.dtknz)
  assert lmp.tknzr._bpe.BPETknzr.dtknz != BaseTknzr.dtknz

  assert inspect.signature(lmp.tknzr._bpe.BPETknzr.tknz) == inspect.signature(BaseTknzr.tknz)
  assert lmp.tknzr._bpe.BPETknzr.tknz != BaseTknzr.tknz
