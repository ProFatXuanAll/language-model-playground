"""Test :py:mod:`lmp.tknzr._base` signatures."""

import argparse
import inspect
import re
from inspect import Parameter, Signature
from typing import Any, ClassVar, Iterable, List, get_type_hints

import lmp.tknzr._base


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.tknzr._base, 'BOS_TK')
  assert lmp.tknzr._base.BOS_TK == '[bos]'
  assert hasattr(lmp.tknzr._base, 'BOS_TKID')
  assert lmp.tknzr._base.BOS_TKID == 0
  assert hasattr(lmp.tknzr._base, 'EOS_TK')
  assert lmp.tknzr._base.EOS_TK == '[eos]'
  assert hasattr(lmp.tknzr._base, 'EOS_TKID')
  assert lmp.tknzr._base.EOS_TKID == 1
  assert hasattr(lmp.tknzr._base, 'PAD_TK')
  assert lmp.tknzr._base.PAD_TK == '[pad]'
  assert hasattr(lmp.tknzr._base, 'PAD_TKID')
  assert lmp.tknzr._base.PAD_TKID == 2
  assert hasattr(lmp.tknzr._base, 'UNK_TK')
  assert lmp.tknzr._base.UNK_TK == '[unk]'
  assert hasattr(lmp.tknzr._base, 'UNK_TKID')
  assert lmp.tknzr._base.UNK_TKID == 3
  assert hasattr(lmp.tknzr._base, 'SP_TKS')
  assert lmp.tknzr._base.SP_TKS == [
    lmp.tknzr._base.BOS_TK,
    lmp.tknzr._base.EOS_TK,
    lmp.tknzr._base.PAD_TK,
    lmp.tknzr._base.UNK_TK,
  ]
  assert hasattr(lmp.tknzr._base, 'WS_PTTN')
  assert isinstance(lmp.tknzr._base.WS_PTTN, re.Pattern)
  assert lmp.tknzr._base.WS_PTTN.pattern == r'\s+'
  assert hasattr(lmp.tknzr._base, 'BaseTknzr')
  assert inspect.isclass(lmp.tknzr._base.BaseTknzr)
  assert inspect.isabstract(lmp.tknzr._base.BaseTknzr)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(lmp.tknzr._base.BaseTknzr) == {'tknzr_name': ClassVar[str]}


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'add_CLI_args')
  assert inspect.ismethod(lmp.tknzr._base.BaseTknzr.add_CLI_args)
  assert lmp.tknzr._base.BaseTknzr.add_CLI_args.__self__ == lmp.tknzr._base.BaseTknzr
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.add_CLI_args) == Signature(
    parameters=[
      Parameter(
        name='parser',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=argparse.ArgumentParser,
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
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='is_uncased',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=bool,
      ),
      Parameter(
        name='max_seq_len',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='max_vocab',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='min_count',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='kwargs',
        kind=Parameter.VAR_KEYWORD,
        annotation=Any,
      ),
    ],
    return_annotation=Signature.empty,
  )
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'batch_dec')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.batch_dec)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.batch_dec) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='batch_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[List[int]],
      ),
      Parameter(
        name='rm_sp_tks',
        kind=Parameter.KEYWORD_ONLY,
        default=False,
        annotation=bool,
      ),
    ],
    return_annotation=List[str],
  )
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'batch_enc')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.batch_enc)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.batch_enc) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='batch_txt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[str],
      ),
    ],
    return_annotation=List[List[int]],
  )
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'build_vocab')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.build_vocab)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.build_vocab) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='batch_txt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=Iterable[str],
      ),
    ],
    return_annotation=None,
  )
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'dec')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.dec)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.dec) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[int],
      ),
      Parameter(
        name='rm_sp_tks',
        kind=Parameter.KEYWORD_ONLY,
        default=False,
        annotation=bool,
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
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='tks',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[str],
      ),
    ],
    return_annotation=str,
  )
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'enc')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.enc)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.enc) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='txt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=List[int],
  )
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'norm')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.norm)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.norm) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='txt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=str,
  )
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'pad_to_max')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.pad_to_max)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.pad_to_max) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[int],
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
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='txt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=List[str],
  )
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'trunc_to_max')
  assert inspect.isfunction(lmp.tknzr._base.BaseTknzr.trunc_to_max)
  assert inspect.signature(lmp.tknzr._base.BaseTknzr.trunc_to_max) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[int],
      ),
    ],
    return_annotation=List[int],
  )
  assert hasattr(lmp.tknzr._base.BaseTknzr, 'vocab_size')
  assert isinstance(lmp.tknzr._base.BaseTknzr.vocab_size, property)
