"""Test :py:mod:`lmp.util.model` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, List

import lmp.util.model
from lmp.model import BaseModel


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.model, 'create')
  assert inspect.isfunction(lmp.util.model.create)
  assert inspect.signature(lmp.util.model.create) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='model_name',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=BaseModel,
  )

  assert hasattr(lmp.util.model, 'list_ckpts')
  assert inspect.isfunction(lmp.util.model.list_ckpts)
  assert inspect.signature(lmp.util.model.list_ckpts) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='exp_name',
      ),
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='first_ckpt',
      ),
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='last_ckpt',
      ),
    ],
    return_annotation=List[int],
  )

  assert hasattr(lmp.util.model, 'load')
  assert inspect.isfunction(lmp.util.model.load)
  assert inspect.signature(lmp.util.model.load) == Signature(
    parameters=[
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='ckpt',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='exp_name',
      ),
    ],
    return_annotation=BaseModel,
  )

  assert hasattr(lmp.util.model, 'save')
  assert inspect.isfunction(lmp.util.model.save)
  assert inspect.signature(lmp.util.model.save) == Signature(
    parameters=[
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='ckpt',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='exp_name',
      ),
      Parameter(
        annotation=BaseModel,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='model',
      ),
    ],
    return_annotation=None,
  )
