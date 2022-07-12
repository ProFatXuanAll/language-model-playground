"""Test :py:mod:`lmp.dset._wnli` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Optional, get_type_hints

import lmp.dset._wnli
from lmp.dset._base import BaseDset


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.dset._wnli, 'WNLIDset')
  assert inspect.isclass(lmp.dset._wnli.WNLIDset)
  assert not inspect.isabstract(lmp.dset._wnli.WNLIDset)
  assert issubclass(lmp.dset._wnli.WNLIDset, BaseDset)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(lmp.dset._wnli.WNLIDset) == get_type_hints(BaseDset)
  assert lmp.dset._wnli.WNLIDset.df_ver == 'train'
  assert lmp.dset._wnli.WNLIDset.dset_name == 'WNLI'
  assert lmp.dset._wnli.WNLIDset.vers == ['dev', 'test', 'train']


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are the same as base class."""
  assert lmp.dset._wnli.WNLIDset.__getitem__ == BaseDset.__getitem__
  assert lmp.dset._wnli.WNLIDset.__iter__ == BaseDset.__iter__
  assert lmp.dset._wnli.WNLIDset.__len__ == BaseDset.__len__


def test_inherent_static_method() -> None:
  """Ensure inherent static methods are the same as base class."""
  assert lmp.dset._wnli.WNLIDset.download_file == BaseDset.download_file
  assert lmp.dset._wnli.WNLIDset.norm == BaseDset.norm


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.dset._wnli.WNLIDset, 'download_dataset')
  assert inspect.ismethod(lmp.dset._wnli.WNLIDset.download_dataset)
  assert lmp.dset._wnli.WNLIDset.download_dataset.__self__ == lmp.dset._wnli.WNLIDset
  assert inspect.signature(lmp.dset._wnli.WNLIDset.download_dataset) == Signature(
    parameters=[],
    return_annotation=None,
  )


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.dset._wnli.WNLIDset, '__init__')
  assert inspect.signature(lmp.dset._wnli.WNLIDset.__init__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='ver',
        kind=Parameter.KEYWORD_ONLY,
        annotation=Optional[str],
        default=None,
      ),
    ],
    return_annotation=Signature.empty,
  )
