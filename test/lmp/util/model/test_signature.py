"""Test :py:mod:`lmp.util.model` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, List, Optional

import lmp.util.model
from lmp.model import BaseModel


def test_module_function() -> None:
  """Ensure module function's signatures."""
  assert inspect.isfunction(lmp.util.model.create)
  assert inspect.signature(lmp.util.model.create) == Signature(
    parameters=[
      Parameter(
        name='model_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='kwargs',
        kind=Parameter.VAR_KEYWORD,
        annotation=Optional[Dict],
      ),
    ],
    return_annotation=BaseModel,
  )

  assert inspect.isfunction(lmp.util.model.load)
  assert inspect.signature(lmp.util.model.load) == Signature(
    parameters=[
      Parameter(
        name='ckpt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='exp_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='model_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='kwargs',
        kind=Parameter.VAR_KEYWORD,
        annotation=Optional[Dict],
      ),
    ],
    return_annotation=BaseModel,
  )

  assert inspect.isfunction(lmp.util.model.list_ckpts)
  assert inspect.signature(lmp.util.model.list_ckpts) == Signature(
    parameters=[
      Parameter(
        name='exp_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='first_ckpt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='last_ckpt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
      ),
    ],
    return_annotation=List[int],
  )
