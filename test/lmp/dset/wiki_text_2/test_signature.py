"""Test :py:class:`lmp.dset.WikiText2Dset` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Optional, get_type_hints

from lmp.dset import BaseDset, WikiText2Dset


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(WikiText2Dset)
  assert not inspect.isabstract(WikiText2Dset)
  assert issubclass(WikiText2Dset, BaseDset)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(WikiText2Dset) == get_type_hints(BaseDset)
  assert WikiText2Dset.df_ver == 'train'
  assert WikiText2Dset.dset_name == 'wiki-text-2'
  assert WikiText2Dset.vers == ['test', 'train', 'valid']


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are the same as base class."""
  assert WikiText2Dset.__getitem__ == BaseDset.__getitem__
  assert WikiText2Dset.__iter__ == BaseDset.__iter__
  assert WikiText2Dset.__len__ == BaseDset.__len__


def test_inherent_static_method() -> None:
  """Ensure inherent static methods are the same as base class."""
  assert WikiText2Dset.norm == BaseDset.norm


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(WikiText2Dset, 'download_dataset')
  assert inspect.ismethod(WikiText2Dset.download_dataset)
  assert WikiText2Dset.download_dataset.__self__ == WikiText2Dset
  assert inspect.signature(WikiText2Dset.download_dataset) == Signature(
    parameters=[],
    return_annotation=None,
  )


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(WikiText2Dset, '__init__')
  assert inspect.signature(WikiText2Dset.__init__) == Signature(
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
