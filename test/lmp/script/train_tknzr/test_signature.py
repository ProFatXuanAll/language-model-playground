"""Test :py:mod:`lmp.script.train_tknzr` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import List

import lmp.script.train_tknzr


def test_module_function() -> None:
  """Ensure module function's signatures."""
  assert inspect.isfunction(lmp.script.train_tknzr.parse_args)
  assert inspect.signature(lmp.script.train_tknzr.parse_args) == Signature(
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
  assert inspect.isfunction(lmp.script.train_tknzr.main)
  assert inspect.signature(lmp.script.train_tknzr.main) == Signature(
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
