"""Test :py:mod:`lmp.infer._top_1` signatures."""

import inspect
from typing import get_type_hints

import lmp.infer._top_1
from lmp.infer._base import BaseInfer


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.infer._top_1, 'Top1Infer')
  assert inspect.isclass(lmp.infer._top_1.Top1Infer)
  assert issubclass(lmp.infer._top_1.Top1Infer, BaseInfer)
  assert not inspect.isabstract(lmp.infer._top_1.Top1Infer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  print(get_type_hints(lmp.infer._top_1.Top1Infer))
  assert get_type_hints(lmp.infer._top_1.Top1Infer) == get_type_hints(BaseInfer)
  assert lmp.infer._top_1.Top1Infer.infer_name == 'top-1'


def test_inherent_class_method():
  """Ensure inherent class methods are the same as base class."""
  assert inspect.signature(lmp.infer._top_1.Top1Infer.add_CLI_args) == inspect.signature(BaseInfer.add_CLI_args)


def test_inherent_instance_method():
  """Ensure inherent instance methods are the same as base class."""
  assert inspect.signature(lmp.infer._top_1.Top1Infer.__init__) == inspect.signature(BaseInfer.__init__)
  assert inspect.signature(lmp.infer._top_1.Top1Infer.gen) == inspect.signature(BaseInfer.gen)
