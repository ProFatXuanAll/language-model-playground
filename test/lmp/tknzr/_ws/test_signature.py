r"""Test :py:class:`lmp.tknzr.WsTknzr` signature."""

import inspect
from inspect import Parameter, Signature
from typing import List, Sequence

from lmp.tknzr._base import BaseTknzr
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


def test_inherent_method():
    r'''Ensure inherent methods are same as baseclass.'''
    assert (
        inspect.signature(BaseTknzr.__init__)
        ==
        inspect.signature(WsTknzr.__init__)
    )

    assert (
        inspect.signature(BaseTknzr.save)
        ==
        inspect.signature(WsTknzr.save)
    )

    assert (
        inspect.signature(BaseTknzr.load)
        ==
        inspect.signature(WsTknzr.load)
    )

    assert (
        inspect.signature(BaseTknzr.norm)
        ==
        inspect.signature(WsTknzr.norm)
    )

    assert (
        inspect.signature(BaseTknzr.tknz)
        ==
        inspect.signature(WsTknzr.tknz)
    )

    assert (
        inspect.signature(BaseTknzr.dtknz)
        ==
        inspect.signature(WsTknzr.dtknz)
    )

    assert (
        inspect.signature(BaseTknzr.enc)
        ==
        inspect.signature(WsTknzr.enc)
    )

    assert (
        inspect.signature(BaseTknzr.dec)
        ==
        inspect.signature(WsTknzr.dec)
    )

    assert (
        inspect.signature(BaseTknzr.batch_enc)
        ==
        inspect.signature(WsTknzr.batch_enc)
    )

    assert (
        inspect.signature(BaseTknzr.batch_dec)
        ==
        inspect.signature(WsTknzr.batch_dec)
    )

    assert (
        inspect.signature(BaseTknzr.build_vocab)
        ==
        inspect.signature(WsTknzr.build_vocab)
    )

    assert (
        inspect.signature(BaseTknzr.train_parser)
        ==
        inspect.signature(WsTknzr.train_parser)
    )
