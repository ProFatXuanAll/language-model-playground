"""Test :py:mod:`lmp.util.dset` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, Iterator, get_type_hints

import torch
import torch.utils.data

import lmp.util.dset
from lmp.dset import BaseDset
from lmp.tknzr import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.util.dset, 'FastTensorDset')
  assert inspect.isclass(lmp.util.dset.FastTensorDset)
  assert hasattr(lmp.util.dset, 'SlowTensorDset')
  assert inspect.isclass(lmp.util.dset.SlowTensorDset)


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.dset, 'load')
  assert inspect.isfunction(lmp.util.dset.load)
  assert inspect.signature(lmp.util.dset.load) == Signature(
    parameters=[
      Parameter(
        name='dset_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='ver',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='kwargs',
        kind=Parameter.VAR_KEYWORD,
        default=Parameter.empty,
        annotation=Any,
      ),
    ],
    return_annotation=BaseDset,
  )


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert get_type_hints(lmp.util.dset.FastTensorDset) == get_type_hints(torch.utils.data.Dataset)
  assert get_type_hints(lmp.util.dset.SlowTensorDset) == get_type_hints(torch.utils.data.Dataset)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.util.dset.FastTensorDset, '__getitem__')
  assert inspect.isfunction(lmp.util.dset.FastTensorDset.__getitem__)
  assert inspect.signature(lmp.util.dset.FastTensorDset.__getitem__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='idx',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=int,
      ),
    ],
    return_annotation=torch.Tensor,
  )
  assert hasattr(lmp.util.dset.FastTensorDset, '__init__')
  assert inspect.isfunction(lmp.util.dset.FastTensorDset.__init__)
  assert inspect.signature(lmp.util.dset.FastTensorDset.__init__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='dset',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=BaseDset,
      ),
      Parameter(
        name='max_seq_len',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='tknzr',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=BaseTknzr,
      ),
    ],
    return_annotation=Signature.empty,
  )
  assert hasattr(lmp.util.dset.FastTensorDset, '__iter__')
  assert inspect.isfunction(lmp.util.dset.FastTensorDset.__iter__)
  assert inspect.signature(lmp.util.dset.FastTensorDset.__iter__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
    ],
    return_annotation=Iterator[torch.Tensor],
  )
  assert hasattr(lmp.util.dset.FastTensorDset, '__len__')
  assert inspect.isfunction(lmp.util.dset.FastTensorDset.__len__)
  assert inspect.signature(lmp.util.dset.FastTensorDset.__len__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
    ],
    return_annotation=int,
  )
  assert hasattr(lmp.util.dset.SlowTensorDset, '__getitem__')
  assert inspect.isfunction(lmp.util.dset.SlowTensorDset.__getitem__)
  assert inspect.signature(lmp.util.dset.SlowTensorDset.__getitem__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='idx',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=int,
      ),
    ],
    return_annotation=torch.Tensor,
  )
  assert hasattr(lmp.util.dset.SlowTensorDset, '__init__')
  assert inspect.isfunction(lmp.util.dset.SlowTensorDset.__init__)
  assert inspect.signature(lmp.util.dset.SlowTensorDset.__init__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='dset',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=BaseDset,
      ),
      Parameter(
        name='max_seq_len',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='tknzr',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=BaseTknzr,
      ),
    ],
    return_annotation=Signature.empty,
  )
  assert hasattr(lmp.util.dset.SlowTensorDset, '__iter__')
  assert inspect.isfunction(lmp.util.dset.SlowTensorDset.__iter__)
  assert inspect.signature(lmp.util.dset.SlowTensorDset.__iter__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
    ],
    return_annotation=Iterator[torch.Tensor],
  )
  assert hasattr(lmp.util.dset.SlowTensorDset, '__len__')
  assert inspect.isfunction(lmp.util.dset.SlowTensorDset.__len__)
  assert inspect.signature(lmp.util.dset.SlowTensorDset.__len__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
    ],
    return_annotation=int,
  )
