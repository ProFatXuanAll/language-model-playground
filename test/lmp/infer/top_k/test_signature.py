"""Test :py:mod:`lmp.infer._top_k` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, get_type_hints

import lmp.infer._top_k
from lmp.infer._base import BaseInfer


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.infer._top_k, 'TopKInfer')
  assert inspect.isclass(lmp.infer._top_k.TopKInfer)
  assert issubclass(lmp.infer._top_k.TopKInfer, BaseInfer)
  assert not inspect.isabstract(lmp.infer._top_k.TopKInfer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  print(get_type_hints(lmp.infer._top_k.TopKInfer))
  assert get_type_hints(lmp.infer._top_k.TopKInfer) == get_type_hints(BaseInfer)
  assert lmp.infer._top_k.TopKInfer.infer_name == 'top-K'


def test_inherent_class_method():
  """Ensure inherent class methods are the same as base class."""
  assert inspect.signature(lmp.infer._top_k.TopKInfer.add_CLI_args) == inspect.signature(BaseInfer.add_CLI_args)


def test_inherent_instance_method():
  """Ensure inherent instance methods are the same as base class."""
  assert inspect.signature(lmp.infer._top_k.TopKInfer.gen) == inspect.signature(BaseInfer.gen)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert inspect.signature(lmp.infer._top_k.TopKInfer.__init__) == Signature(
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
