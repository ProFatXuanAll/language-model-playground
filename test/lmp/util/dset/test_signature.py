r"""Test :py:mod:`lmp.util.dset` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Optional

import lmp.util.dset
from lmp.dset import BaseDset


def test_module_function():
    """Ensure module function's signature."""
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
                default=None,
                annotation=Optional[str],
            ),
        ],
        return_annotation=BaseDset,
    )
