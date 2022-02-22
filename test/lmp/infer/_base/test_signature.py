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
  assert hasattr(lmp.infer._base.BaseInfer, '__init__')
  assert inspect.signature(lmp.infer._base.BaseInfer.__init__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='max_seq_len',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
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
  assert hasattr(lmp.infer._base.BaseInfer, 'gen')
  assert 'gen' in lmp.infer._base.BaseInfer.__abstractmethods__
  assert inspect.signature(lmp.infer._base.BaseInfer.gen) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='model',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=BaseModel,
      ),
      Parameter(
        name='tknzr',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=BaseTknzr,
      ),
      Parameter(
        name='txt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=str,
      ),
    ],
    return_annotation=str,
  )
