"""Test :py:mod:`lmp.util.infer` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any

import lmp.util.infer
from lmp.infer import BaseInfer


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.infer, 'create')
  assert inspect.isfunction(lmp.util.infer.create)
  assert inspect.signature(lmp.util.infer.create) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='infer_name',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=BaseInfer,
  )
