"""Test :py:class:`lmp.infer.BaseInfer` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, get_type_hints

from lmp.infer import BaseInfer
from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


def test_class() -> None:
  """Ensure abstract class signatures."""
  assert inspect.isclass(BaseInfer)
  assert inspect.isabstract(BaseInfer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  print(get_type_hints(BaseInfer))
  assert get_type_hints(BaseInfer) == {
    'infer_name': ClassVar[str],
  }
  assert BaseInfer.infer_name == 'base'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(BaseInfer, 'infer_parser')
  assert inspect.ismethod(BaseInfer.infer_parser)
  assert BaseInfer.infer_parser.__self__ == BaseInfer
  assert inspect.signature(BaseInfer.infer_parser) == Signature(
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
  assert hasattr(BaseInfer, '__init__')
  assert inspect.signature(BaseInfer.__init__) == Signature(
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
  assert hasattr(BaseInfer, 'gen')
  assert 'gen' in BaseInfer.__abstractmethods__
  assert inspect.signature(BaseInfer.gen) == Signature(
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
