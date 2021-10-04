r"""Test :py:mod:`lmp.util.infer` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

import lmp.util.infer
from lmp.infer import BaseInfer


def test_module_function():
    """Ensure module function's signature."""
    assert inspect.isfunction(lmp.util.infer.create)
    assert inspect.signature(lmp.util.infer.create) == Signature(
        parameters=[
            Parameter(
                name='infer_name',
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
        return_annotation=BaseInfer,
    )
