"""Test :py:mod:`lmp.tknzr._base` signatures."""

import argparse
import inspect
import re
from inspect import Parameter, Signature
from typing import Any, ClassVar, Iterable, List, get_type_hints

import lmp.tknzr._base


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.tknzr._base, 'WS_PTTN')
  assert isinstance(lmp.tknzr._base.WS_PTTN, re.Pattern)
  assert lmp.tknzr._base.WS_PTTN.pattern == r'\s+'

  assert hasattr(lmp.tknzr._base, 'BaseTknzr')
  assert inspect.isclass(lmp.tknzr._base.BaseTknzr)
  assert inspect.isabstract(lmp.tknzr._base.BaseTknzr)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(lmp.tknzr._base.BaseTknzr) == {'tknzr_name': ClassVar[str]}
  assert lmp.tknzr._base.BaseTknzr.tknzr_name == 'base'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'add_CLI_args')
  assert inspect.ismethod(lmp.tknzr._base.BaseTknzr.add_CLI_args)
  assert lmp.tknzr._base.BaseTknzr.add_CLI_args.__self__ == lmp.tknzr._base.BaseTknzr
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.add_CLI_args) == Signature(
    parameters=[
      Parameter(
        annotation=argparse.ArgumentParser,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='parser',
      ),
    ],
    return_annotation=None,
  )


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.tknzr._base.BaseTknzr, '__init__')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.__init__)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.__init__) == Signature(
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
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.tknzr._base.BaseTknzr, 'build_vocab')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.build_vocab)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.build_vocab) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=Iterable[str],
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_txt',
      ),
    ],
    return_annotation=None,
  )

  assert hasattr(lmp.tknzr._base.BaseTknzr, 'dec')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.dec)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.dec) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=List[int],
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='tkids',
      ),
      Parameter(
        annotation=bool,
        default=False,
        kind=Parameter.KEYWORD_ONLY,
        name='rm_sp_tks',
      ),
    ],
    return_annotation=str,
  )

  assert hasattr(lmp.tknzr._base.BaseTknzr, 'dtknz')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.dtknz)
  assert 'dtknz' in lmp.tknzr._base.BaseTknzr.__abstractmethods__
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.dtknz) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=List[str],
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='tks',
      ),
    ],
    return_annotation=str,
  )

  assert hasattr(lmp.tknzr._base.BaseTknzr, 'enc')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.enc)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.enc) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='txt',
      ),
    ],
    return_annotation=List[int],
  )

  assert hasattr(lmp.tknzr._base.BaseTknzr, 'norm')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.norm)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.norm) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='txt',
      ),
    ],
    return_annotation=str,
  )

  assert hasattr(lmp.tknzr._base.BaseTknzr, 'pad_to_max')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.pad_to_max)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.pad_to_max) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='max_seq_len',
      ),
      Parameter(
        annotation=List[int],
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='tkids',
      ),
    ],
    return_annotation=List[int],
  )

  assert hasattr(lmp.tknzr._base.BaseTknzr, 'tknz')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.tknz)
  assert 'tknz' in lmp.tknzr._base.BaseTknzr.__abstractmethods__
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.tknz) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='txt',
      ),
    ],
    return_annotation=List[str],
  )

  assert hasattr(lmp.tknzr._base.BaseTknzr, 'vocab_size')
  assert isinstance(lmp.tknzr._base.BaseTknzr.vocab_size, property)
