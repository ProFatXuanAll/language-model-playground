"""Test :py:mod:`lmp.script.train_tknzr` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import List

import lmp.script.train_tknzr


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.script.train_tknzr, 'parse_args')
  assert inspect.isfunction(lmp.script.train_tknzr.parse_args)
  assert inspect.signature(lmp.script.train_tknzr.parse_args) == Signature(
    parameters=[
      Parameter(
        annotation=List[str],
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='argv',
      ),
    ],
    return_annotation=argparse.Namespace,
  )

  assert hasattr(lmp.script.train_tknzr, 'main')
  assert inspect.isfunction(lmp.script.train_tknzr.main)
  assert inspect.signature(lmp.script.train_tknzr.main) == Signature(
    parameters=[
      Parameter(
        annotation=List[str],
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='argv',
      ),
    ],
    return_annotation=None,
  )
