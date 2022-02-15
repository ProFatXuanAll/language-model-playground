"""Test :py:mod:`lmp.util.tknzr` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any

import lmp.util.tknzr
from lmp.tknzr import BaseTknzr


def test_module_function() -> None:
  """Ensure module function's signatures."""
  assert inspect.isfunction(lmp.util.tknzr.create)
  assert inspect.signature(lmp.util.tknzr.create) == Signature(
    parameters=[
      Parameter(
        name='tknzr_name',
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
    return_annotation=BaseTknzr,
  )
  assert inspect.isfunction(lmp.util.tknzr.load)
  assert inspect.signature(lmp.util.tknzr.load) == Signature(
    parameters=[
      Parameter(
        name='exp_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=BaseTknzr,
  )
  assert inspect.isfunction(lmp.util.tknzr.save)
  assert inspect.signature(lmp.util.tknzr.save) == Signature(
    parameters=[
      Parameter(
        name='exp_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='tknzr',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=BaseTknzr,
      ),
    ],
    return_annotation=None,
  )
