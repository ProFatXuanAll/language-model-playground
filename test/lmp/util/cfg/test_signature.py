r"""Test :py:mod:`lmp.util.cfg` signature."""

import argparse
import inspect
from inspect import Parameter, Signature

import lmp.util.cfg


def test_module_attribute():
    """Ensure module attribute's signature."""
    assert lmp.util.cfg.CFG_NAME == 'cfg.json'


def test_module_function():
    """Ensure module function's signature."""
    assert inspect.isfunction(lmp.util.cfg.load)
    assert inspect.signature(lmp.util.cfg.load) == Signature(
        parameters=[
            Parameter(
                name='exp_name',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ],
        return_annotation=argparse.Namespace,
    )

    assert inspect.isfunction(lmp.util.cfg.save)
    assert inspect.signature(lmp.util.cfg.save) == Signature(
        parameters=[
            Parameter(
                name='args',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=argparse.Namespace,
            ),
            Parameter(
                name='exp_name',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ],
        return_annotation=None,
    )
