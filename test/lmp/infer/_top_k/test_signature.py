r"""Test :py:class:`lmp.infer.Top1Infer` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional, get_type_hints

from lmp.infer import BaseInfer, TopKInfer


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(TopKInfer)
    assert not inspect.isabstract(TopKInfer)
    assert issubclass(TopKInfer, BaseInfer)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    print(get_type_hints(TopKInfer))
    assert get_type_hints(TopKInfer) == get_type_hints(BaseInfer)
    assert TopKInfer.hard_max_seq_len == BaseInfer.hard_max_seq_len
    assert TopKInfer.infer_name == 'top-k'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert inspect.signature(TopKInfer.__init__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='k',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=int,
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
        inspect.signature(TopKInfer.gen)
        ==
        inspect.signature(BaseInfer.gen)
    )
    assert (
        inspect.signature(TopKInfer.infer_parser)
        ==
        inspect.signature(BaseInfer.infer_parser)
    )
