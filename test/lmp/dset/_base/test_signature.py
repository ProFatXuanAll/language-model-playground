"""Test :py:mod:`lmp.dset._base` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import ClassVar, Iterator, List, Optional, get_type_hints

import torch.utils.data

import lmp.dset._base


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.dset._base, 'BaseDset')
  assert inspect.isclass(lmp.dset._base.BaseDset)
  assert not inspect.isabstract(lmp.dset._base.BaseDset)
  assert issubclass(lmp.dset._base.BaseDset, torch.utils.data.Dataset)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(lmp.dset._base.BaseDset)
  assert type_hints['df_ver'] == ClassVar[str]
  assert type_hints['dset_name'] == ClassVar[str]
  assert type_hints['vers'] == ClassVar[List[str]]
  assert lmp.dset._base.BaseDset.df_ver == ''
  assert lmp.dset._base.BaseDset.dset_name == 'base'
  assert lmp.dset._base.BaseDset.vers == []


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.dset._base.BaseDset, '__getitem__')
  assert inspect.isfunction(lmp.dset._base.BaseDset.__getitem__)
  assert inspect.signature(lmp.dset._base.BaseDset.__getitem__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='idx',
      )
    ],
    return_annotation=str,
  )

  assert hasattr(lmp.dset._base.BaseDset, '__init__')
  assert inspect.isfunction(lmp.dset._base.BaseDset.__init__)
  assert inspect.signature(lmp.dset._base.BaseDset.__init__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=Optional[str],
        default=None,
        kind=Parameter.KEYWORD_ONLY,
        name='ver',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.dset._base.BaseDset, '__iter__')
  assert inspect.isfunction(lmp.dset._base.BaseDset.__iter__)
  assert inspect.signature(lmp.dset._base.BaseDset.__iter__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
    ],
    return_annotation=Iterator[str],
  )

  assert hasattr(lmp.dset._base.BaseDset, '__len__')
  assert inspect.isfunction(lmp.dset._base.BaseDset.__len__)
  assert inspect.signature(lmp.dset._base.BaseDset.__len__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
    ],
    return_annotation=int,
  )


def test_static_method() -> None:
  """Ensure static methods' signatures."""
  assert hasattr(lmp.dset._base.BaseDset, 'download_file')
  assert inspect.isfunction(lmp.dset._base.BaseDset.download_file)
  assert inspect.signature(lmp.dset._base.BaseDset.download_file) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='mode',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='download_path',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='url',
      ),
    ],
    return_annotation=None,
  )

  assert hasattr(lmp.dset._base.BaseDset, 'norm')
  assert inspect.isfunction(lmp.dset._base.BaseDset.norm)
  assert inspect.signature(lmp.dset._base.BaseDset.norm) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='txt',
      ),
    ],
    return_annotation=str,
  )
