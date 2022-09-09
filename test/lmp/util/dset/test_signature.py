"""Test :py:mod:`lmp.util.dset` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, Tuple

import torch

import lmp.util.dset
from lmp.dset import BaseDset
from lmp.tknzr import BaseTknzr


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.dset, 'LMFormatDset')
  assert inspect.isclass(lmp.util.dset.LMFormatDset)

  assert hasattr(lmp.util.dset.LMFormatDset, '__init__')
  assert inspect.isfunction(lmp.util.dset.LMFormatDset.__init__)
  assert inspect.signature(lmp.util.dset.LMFormatDset.__init__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=BaseDset,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='dset',
      ),
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='max_seq_len',
      ),
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='stride',
      ),
      Parameter(
        annotation=BaseTknzr,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='tknzr',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.util.dset.LMFormatDset, '__getitem__')
  assert inspect.isfunction(lmp.util.dset.LMFormatDset.__getitem__)
  assert inspect.signature(lmp.util.dset.LMFormatDset.__getitem__) == Signature(
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
      ),
    ],
    return_annotation=Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
  )

  assert hasattr(lmp.util.dset.LMFormatDset, '__len__')
  assert inspect.isfunction(lmp.util.dset.LMFormatDset.__len__)
  assert inspect.signature(lmp.util.dset.LMFormatDset.__len__) == Signature(
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

  assert hasattr(lmp.util.dset, 'load')
  assert inspect.isfunction(lmp.util.dset.load)
  assert inspect.signature(lmp.util.dset.load) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='dset_name',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='ver',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=BaseDset,
  )
