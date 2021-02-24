r"""Test :py:class:`lmp.tknzr.CharTknzr` signature."""

import inspect
from inspect import Parameter, Signature
from typing import (List, Sequence)

from lmp.tknzr._char import CharTknzr
from lmp.tknzr._base import BaseTknzr


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


def test_inherent_method():
    r'''Ensure inherent methods are same as baseclass.'''
    assert inspect.signature(
        BaseTknzr.__init__) == inspect.signature(
        CharTknzr.__init__)

    assert inspect.signature(
        BaseTknzr.save) == inspect.signature(
        CharTknzr.save)

    assert inspect.signature(
        BaseTknzr.load) == inspect.signature(
        CharTknzr.load)

    assert inspect.signature(
        BaseTknzr.norm) == inspect.signature(
        CharTknzr.norm)

    assert inspect.signature(
        BaseTknzr.tknz) == inspect.signature(
        CharTknzr.tknz)

    assert inspect.signature(
        BaseTknzr.dtknz) == inspect.signature(
        CharTknzr.dtknz)

    assert inspect.signature(
        BaseTknzr.enc) == inspect.signature(
        CharTknzr.enc)

    assert inspect.signature(
        BaseTknzr.dec) == inspect.signature(
        CharTknzr.dec)

    assert inspect.signature(
        BaseTknzr.batch_enc) == inspect.signature(
        CharTknzr.batch_enc)

    assert inspect.signature(
        BaseTknzr.batch_dec) == inspect.signature(
        CharTknzr.batch_dec)

    assert inspect.signature(
        BaseTknzr.build_vocab) == inspect.signature(
        CharTknzr.build_vocab)

    assert inspect.signature(
        BaseTknzr.train_parser) == inspect.signature(
        CharTknzr.train_parser)
