"""Test :py:mod:`lmp.util.dset` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any

import lmp.util.dset
from lmp.dset import BaseDset


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
