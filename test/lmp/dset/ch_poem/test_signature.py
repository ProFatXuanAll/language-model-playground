"""Test :py:class:`lmp.dset.ChPoemDset` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Optional, get_type_hints

from lmp.dset import BaseDset, ChPoemDset


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(ChPoemDset)
  assert not inspect.isabstract(ChPoemDset)
  assert issubclass(ChPoemDset, BaseDset)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(ChPoemDset) == get_type_hints(BaseDset)
  assert ChPoemDset.df_ver == '唐'
  assert ChPoemDset.dset_name == 'chinese-poem'
  assert ChPoemDset.vers == [
    '元',
    '元末明初',
    '先秦',
    '南北朝',
    '唐',
    '唐末宋初',
    '宋',
    '宋末元初',
    '宋末金初',
    '明',
    '明末清初',
    '民國末當代初',
    '清',
    '清末民國初',
    '清末近現代初',
    '漢',
    '當代',
    '秦',
    '近現代',
    '近現代末當代初',
    '遼',
    '金',
    '金末元初',
    '隋',
    '隋末唐初',
    '魏晉',
    '魏晉末南北朝初',
  ]


def test_inherent_instance_method() -> None:
  """Ensure inherent instance methods are the same as base class."""
  assert ChPoemDset.__getitem__ == BaseDset.__getitem__
  assert ChPoemDset.__iter__ == BaseDset.__iter__
  assert ChPoemDset.__len__ == BaseDset.__len__


def test_inherent_static_method() -> None:
  """Ensure inherent static methods are the same as base class."""
  assert ChPoemDset.norm == BaseDset.norm


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(ChPoemDset, 'download_dataset')
  assert inspect.ismethod(ChPoemDset.download_dataset)
  assert ChPoemDset.download_dataset.__self__ == ChPoemDset
  assert inspect.signature(ChPoemDset.download_dataset) == Signature(
    parameters=[
      Parameter(
        name='ver',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=None,
  )


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(ChPoemDset, '__init__')
  assert inspect.signature(ChPoemDset.__init__) == Signature(
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
