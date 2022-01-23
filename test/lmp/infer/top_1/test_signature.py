"""Test :py:class:`lmp.infer.Top1Infer` signatures."""

import inspect
from typing import get_type_hints

from lmp.infer import BaseInfer, Top1Infer


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(Top1Infer)
  assert not inspect.isabstract(Top1Infer)
  assert issubclass(Top1Infer, BaseInfer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  print(get_type_hints(Top1Infer))
  assert get_type_hints(Top1Infer) == get_type_hints(BaseInfer)
  assert Top1Infer.infer_name == 'top-1'


def test_inherent_class_method():
  """Ensure inherent class methods are the same as base class."""
  assert inspect.signature(Top1Infer.infer_parser) == inspect.signature(BaseInfer.infer_parser)


def test_inherent_instance_method():
  """Ensure inherent instance methods are the same as base class."""
  assert inspect.signature(Top1Infer.__init__) == inspect.signature(BaseInfer.__init__)
  assert inspect.signature(Top1Infer.gen) == inspect.signature(BaseInfer.gen)
