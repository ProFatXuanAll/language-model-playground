"""Test :py:mod:`lmp.script.sample_dset` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import List

import lmp.script.sample_dset


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.script.sample_dset, 'parse_args')
  assert inspect.isfunction(lmp.script.sample_dset.parse_args)
  assert inspect.signature(lmp.script.sample_dset.parse_args) == Signature(
    parameters=[
      Parameter(
        name='argv',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[str],
      ),
    ],
    return_annotation=argparse.Namespace,
  )
  assert hasattr(lmp.script.sample_dset, 'main')
  assert inspect.isfunction(lmp.script.sample_dset.main)
  assert inspect.signature(lmp.script.sample_dset.main) == Signature(
    parameters=[
      Parameter(
        name='argv',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[str],
      ),
    ],
    return_annotation=None,
  )
