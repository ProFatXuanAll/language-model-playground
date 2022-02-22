"""Test :py:mod:`lmp.dset._wiki_text_2` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Optional, get_type_hints

import lmp.dset._wiki_text_2
from lmp.dset._base import BaseDset


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.dset._wiki_text_2, 'WikiText2Dset')
  assert inspect.isclass(lmp.dset._wiki_text_2.WikiText2Dset)
  assert not inspect.isabstract(lmp.dset._wiki_text_2.WikiText2Dset)
  assert issubclass(lmp.dset._wiki_text_2.WikiText2Dset, BaseDset)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(lmp.dset._wiki_text_2.WikiText2Dset) == get_type_hints(BaseDset)
  assert lmp.dset._wiki_text_2.WikiText2Dset.df_ver == 'train'
  assert lmp.dset._wiki_text_2.WikiText2Dset.dset_name == 'wiki-text-2'
  assert lmp.dset._wiki_text_2.WikiText2Dset.vers == ['test', 'train', 'valid']


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are the same as base class."""
  assert lmp.dset._wiki_text_2.WikiText2Dset.__getitem__ == BaseDset.__getitem__
  assert lmp.dset._wiki_text_2.WikiText2Dset.__iter__ == BaseDset.__iter__
  assert lmp.dset._wiki_text_2.WikiText2Dset.__len__ == BaseDset.__len__


def test_inherent_static_method() -> None:
  """Ensure inherent static methods are the same as base class."""
  assert lmp.dset._wiki_text_2.WikiText2Dset.norm == BaseDset.norm


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.dset._wiki_text_2.WikiText2Dset, 'download_dataset')
  assert inspect.ismethod(lmp.dset._wiki_text_2.WikiText2Dset.download_dataset)
  assert lmp.dset._wiki_text_2.WikiText2Dset.download_dataset.__self__ == lmp.dset._wiki_text_2.WikiText2Dset
  assert inspect.signature(lmp.dset._wiki_text_2.WikiText2Dset.download_dataset) == Signature(
    parameters=[],
    return_annotation=None,
  )


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.dset._wiki_text_2.WikiText2Dset, '__init__')
  assert inspect.signature(lmp.dset._wiki_text_2.WikiText2Dset.__init__) == Signature(
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
