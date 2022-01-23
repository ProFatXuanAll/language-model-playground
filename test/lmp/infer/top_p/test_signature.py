"""Test :py:class:`lmp.infer.Top1Infer` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, get_type_hints

from lmp.infer import BaseInfer, TopPInfer


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(TopPInfer)
  assert not inspect.isabstract(TopPInfer)
  assert issubclass(TopPInfer, BaseInfer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  print(get_type_hints(TopPInfer))
  assert get_type_hints(TopPInfer) == get_type_hints(BaseInfer)
  assert TopPInfer.infer_name == 'top-p'


def test_inherent_class_method():
  """Ensure inherent class methods are the same as base class."""
  assert inspect.signature(TopPInfer.infer_parser) == inspect.signature(BaseInfer.infer_parser)


def test_inherent_intance_method():
  """Ensure inherent intance methods are the same as base class."""
  assert inspect.signature(TopPInfer.gen) == inspect.signature(BaseInfer.gen)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert inspect.signature(TopPInfer.__init__) == Signature(
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
        name='p',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=float,
      ),
      Parameter(
        name='kwargs',
        kind=Parameter.VAR_KEYWORD,
        annotation=Any,
      ),
    ],
    return_annotation=Signature.empty,
  )
