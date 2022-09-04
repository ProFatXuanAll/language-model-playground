"""Test :py:mod:`lmp.util.tknzr` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any

import lmp.util.tknzr
from lmp.tknzr import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.util.tknzr, 'FILE_NAME')
  assert lmp.util.tknzr.FILE_NAME == 'tknzr.pkl'


def test_module_method() -> None:
  """Ensure module functions' signatures."""
  assert hasattr(lmp.util.tknzr, 'create')
  assert inspect.isfunction(lmp.util.tknzr.create)
  assert inspect.signature(lmp.util.tknzr.create) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='tknzr_name',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=BaseTknzr,
  )

  assert hasattr(lmp.util.tknzr, 'load')
  assert inspect.isfunction(lmp.util.tknzr.load)
  assert inspect.signature(lmp.util.tknzr.load) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='exp_name',
      ),
    ],
    return_annotation=BaseTknzr,
  )

  assert hasattr(lmp.util.tknzr, 'save')
  assert inspect.isfunction(lmp.util.tknzr.save)
  assert inspect.signature(lmp.util.tknzr.save) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='exp_name',
      ),
      Parameter(
        annotation=BaseTknzr,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='tknzr',
      ),
    ],
    return_annotation=None,
  )
