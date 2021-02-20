r"""Test :py:class:`lmp.tknzr.CharTknzr` signature."""

import inspect
from inspect import Parameter, Signature
from typing import (List, Sequence)

from lmp.tknzr._char import CharTknzr


def test_class():
    r"""Ensure abstract class signature.

    Subclass only need to implement method tknzr and dtknzr.
    """
    assert inspect.isclass(CharTknzr)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(CharTknzr.tknzr_name, str)
    assert CharTknzr.tknzr_name == 'character'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(CharTknzr, 'tknz')
    assert inspect.signature(CharTknzr.tknz) == Signature(
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

    assert hasattr(CharTknzr, 'dtknz')
    assert inspect.signature(CharTknzr.dtknz) == Signature(
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
