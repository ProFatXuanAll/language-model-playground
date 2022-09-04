"""Test :py:mod:`lmp.dset._demo` signatures."""

import inspect
from typing import get_type_hints

import lmp.dset._demo
from lmp.dset._base import BaseDset


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.dset._demo, 'DemoDset')
  assert inspect.isclass(lmp.dset._demo.DemoDset)
  assert not inspect.isabstract(lmp.dset._demo.DemoDset)
  assert issubclass(lmp.dset._demo.DemoDset, BaseDset)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(lmp.dset._demo.DemoDset) == get_type_hints(BaseDset)
  assert lmp.dset._demo.DemoDset.df_ver == 'train'
  assert lmp.dset._demo.DemoDset.dset_name == 'demo'
  assert lmp.dset._demo.DemoDset.vers == [
    'test',
    'train',
    'valid',
  ]


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are the same as base class."""
  assert lmp.dset._demo.DemoDset.__getitem__ == BaseDset.__getitem__
  assert lmp.dset._demo.DemoDset.__iter__ == BaseDset.__iter__
  assert lmp.dset._demo.DemoDset.__len__ == BaseDset.__len__


def test_inherent_static_method() -> None:
  """Ensure inherent static methods are the same as base class."""
  assert lmp.dset._demo.DemoDset.download_file == BaseDset.download_file
  assert lmp.dset._demo.DemoDset.norm == BaseDset.norm


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.dset._demo.DemoDset, '__init__')
  assert inspect.signature(lmp.dset._demo.DemoDset.__init__) == inspect.signature(BaseDset.__init__)
