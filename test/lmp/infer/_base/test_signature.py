"""Test :py:mod:`lmp.infer._base` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, get_type_hints

import lmp.infer._base
from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.infer._base, 'BaseInfer')
  assert inspect.isclass(lmp.infer._base.BaseInfer)
  assert inspect.isabstract(lmp.infer._base.BaseInfer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  print(get_type_hints(lmp.infer._base.BaseInfer))
  assert get_type_hints(lmp.infer._base.BaseInfer) == {
    'infer_name': ClassVar[str],
  }
  assert lmp.infer._base.BaseInfer.infer_name == 'base'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.infer._base.BaseInfer, 'add_CLI_args')
  assert inspect.ismethod(lmp.infer._base.BaseInfer.add_CLI_args)
  assert lmp.infer._base.BaseInfer.add_CLI_args.__self__ == lmp.infer._base.BaseInfer
  assert inspect.signature(lmp.infer._base.BaseInfer.add_CLI_args) == Signature(
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
  assert hasattr(lmp.infer._base.BaseInfer, '__init__')
  assert inspect.signature(lmp.infer._base.BaseInfer.__init__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=int,
        default=32,
        kind=Parameter.KEYWORD_ONLY,
        name='max_seq_len',
      ),
      Parameter(
        annotation=Any,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.infer._base.BaseInfer, 'gen')
  assert 'gen' in lmp.infer._base.BaseInfer.__abstractmethods__
  assert inspect.signature(lmp.infer._base.BaseInfer.gen) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=BaseModel,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='model',
      ),
      Parameter(
        annotation=BaseTknzr,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='tknzr',
      ),
      Parameter(
        annotation=str,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='txt',
      ),
    ],
    return_annotation=str,
  )
