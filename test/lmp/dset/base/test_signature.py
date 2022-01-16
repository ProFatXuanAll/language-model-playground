"""Test :py:class:`lmp.dset.BaseDset` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import ClassVar, Iterator, List, Optional, get_type_hints

import torch.utils.data

from lmp.dset import BaseDset


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(BaseDset)
  assert not inspect.isabstract(BaseDset)
  assert issubclass(BaseDset, torch.utils.data.Dataset)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(BaseDset)
  assert type_hints['df_ver'] == ClassVar[str]
  assert type_hints['dset_name'] == ClassVar[str]
  assert type_hints['vers'] == ClassVar[List[str]]
  assert BaseDset.df_ver == ''
  assert BaseDset.dset_name == 'base'
  assert BaseDset.vers == []


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(BaseDset, '__getitem__')
  assert inspect.signature(BaseDset.__getitem__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='idx',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
      )
    ],
    return_annotation=str,
  )
  assert hasattr(BaseDset, '__init__')
  assert inspect.signature(BaseDset.__init__) == Signature(
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
  assert hasattr(BaseDset, '__iter__')
  assert inspect.signature(BaseDset.__iter__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
    ],
    return_annotation=Iterator[str],
  )
  assert hasattr(BaseDset, '__len__')
  assert inspect.signature(BaseDset.__len__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
    ],
    return_annotation=int,
  )


def test_static_method() -> None:
  """Ensure static methods' signatures."""
  assert hasattr(BaseDset, 'download_file')
  assert inspect.isfunction(BaseDset.download_file)
  assert inspect.signature(BaseDset.download_file) == Signature(
    parameters=[
      Parameter(
        name='mode',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='download_path',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='url',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=None,
  )
  assert hasattr(BaseDset, 'norm')
  assert inspect.isfunction(BaseDset.norm)
  assert inspect.signature(BaseDset.norm) == Signature(
    parameters=[
      Parameter(
        name='txt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=str,
  )
