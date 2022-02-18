"""Test :py:mod:`lmp.infer._top_p` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, get_type_hints

import lmp.infer._top_p
from lmp.infer._base import BaseInfer


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.infer._top_p, 'TopPInfer')
  assert inspect.isclass(lmp.infer._top_p.TopPInfer)
  assert not inspect.isabstract(lmp.infer._top_p.TopPInfer)
  assert issubclass(lmp.infer._top_p.TopPInfer, BaseInfer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  print(get_type_hints(lmp.infer._top_p.TopPInfer))
  assert get_type_hints(lmp.infer._top_p.TopPInfer) == get_type_hints(BaseInfer)
  assert lmp.infer._top_p.TopPInfer.infer_name == 'top-P'


def test_inherent_class_method():
  """Ensure inherent class methods are the same as base class."""
  assert inspect.signature(lmp.infer._top_p.TopPInfer.add_CLI_args) == inspect.signature(BaseInfer.add_CLI_args)


def test_inherent_intance_method():
  """Ensure inherent intance methods are the same as base class."""
  assert inspect.signature(lmp.infer._top_p.TopPInfer.gen) == inspect.signature(BaseInfer.gen)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert inspect.signature(lmp.infer._top_p.TopPInfer.__init__) == Signature(
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
