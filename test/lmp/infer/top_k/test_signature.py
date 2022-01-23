"""Test :py:class:`lmp.infer.Top1Infer` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, get_type_hints

from lmp.infer import BaseInfer, TopKInfer


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(TopKInfer)
  assert not inspect.isabstract(TopKInfer)
  assert issubclass(TopKInfer, BaseInfer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  print(get_type_hints(TopKInfer))
  assert get_type_hints(TopKInfer) == get_type_hints(BaseInfer)
  assert TopKInfer.infer_name == 'top-k'


def test_inherent_class_method():
  """Ensure inherent class methods are the same as base class."""
  assert inspect.signature(TopKInfer.infer_parser) == inspect.signature(BaseInfer.infer_parser)


def test_inherent_instance_method():
  """Ensure inherent instance methods are the same as base class."""
  assert inspect.signature(TopKInfer.gen) == inspect.signature(BaseInfer.gen)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert inspect.signature(TopKInfer.__init__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='k',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
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
