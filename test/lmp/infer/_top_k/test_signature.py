r"""Test :py:class:`lmp.infer.Top1Infer` signature."""

import inspect
from inspect import Parameter, Signature
from typing import (ClassVar, get_type_hints)

from lmp.infer._top_k import TopKInfer
from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr
from lmp.infer._base import BaseInfer


def test_class():
    r"""Ensure abstract class signature.

    Subclass only need to implement method gen.
    """
    assert inspect.isclass(TopKInfer)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    print(get_type_hints(TopKInfer))
    assert get_type_hints(TopKInfer) == {
        'hard_max_seq_len': ClassVar[int],
        'infer_name': ClassVar[str],
    }
    assert TopKInfer.infer_name == 'top-k'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(TopKInfer, 'gen')
    assert inspect.signature(TopKInfer.gen) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='model',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=BaseModel,
            ),
            Parameter(
                name='tknzr',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=BaseTknzr,
            ),
            Parameter(
                name='txt',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            ),
        ],
        return_annotation=str,
    )


def test_subclass_method():
    r'''Ensure inherent methods are same as baseclass.'''
    assert inspect.signature(
        BaseInfer.gen) == inspect.signature(
        TopKInfer.gen)

    assert inspect.signature(
        BaseInfer.infer_parser) == inspect.signature(
        TopKInfer.infer_parser)
