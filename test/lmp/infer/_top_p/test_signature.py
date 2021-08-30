r"""Test :py:class:`lmp.infer.Top1Infer` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional, get_type_hints

from lmp.infer import BaseInfer, TopPInfer


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(TopPInfer)
    assert not inspect.isabstract(TopPInfer)
    assert issubclass(TopPInfer, BaseInfer)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    print(get_type_hints(TopPInfer))
    assert get_type_hints(TopPInfer) == get_type_hints(BaseInfer)
    assert TopPInfer.hard_max_seq_len == BaseInfer.hard_max_seq_len
    assert TopPInfer.infer_name == 'top-p'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert inspect.signature(TopPInfer.__init__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='p',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=float,
            ),
            Parameter(
                name='max_seq_len',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=int,
            ),
            Parameter(
                name='kwargs',
                kind=Parameter.VAR_KEYWORD,
                annotation=Optional[Dict],
            ),
        ],
        return_annotation=Signature.empty,
    )


def test_inherent_method():
    r'''Ensure inherent methods' signature are the same as base class.'''
    assert (
        inspect.signature(TopPInfer.gen)
        ==
        inspect.signature(BaseInfer.gen)
    )
    assert (
        inspect.signature(TopPInfer.infer_parser)
        ==
        inspect.signature(BaseInfer.infer_parser)
    )
