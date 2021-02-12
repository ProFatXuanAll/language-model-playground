r"""Test :py:class:`lmp.tknzr.WsTknzr` signature."""

import inspect
from inspect import Parameter, Signature
from typing import (List, Sequence)

from lmp.tknzr._ws import WsTknzr


def test_class():
    r"""Ensure abstract class signature.

    Subclass only need to implement method tknzr and dtknzr.
    """
    assert inspect.isclass(WsTknzr)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(WsTknzr.tknzr_name, str)
    assert WsTknzr.tknzr_name == 'whitespace'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(WsTknzr, 'tknz')
    assert inspect.signature(WsTknzr.tknz) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='txt',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ],
        return_annotation=List[str],
    )

    assert hasattr(WsTknzr, 'dtknz')
    assert inspect.signature(WsTknzr.dtknz) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='tks',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=Sequence[str],
            ),
        ],
        return_annotation=str,
    )
