r"""Test :py:class:`lmp.infer.BaseInfer` signature."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import (ClassVar, Dict, Optional,
                    get_type_hints)

from lmp.infer._base import BaseInfer


def test_class():
    r"""Ensure abstract class signature.

    Subclass only need to implement method tknzr and dtknzr.
    """
    assert inspect.isclass(BaseInfer)
    assert inspect.isabstract(BaseInfer)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    print(get_type_hints(BaseInfer))
    assert get_type_hints(BaseInfer) == {
        'hard_max_seq_len': ClassVar[int],
        'infer_name': ClassVar[str],
    }
    assert BaseInfer.hard_max_seq_len == 512
    assert BaseInfer.infer_name == 'base'


def test_instance_method(subclss_infer):
    r"""Ensure instance methods' signature."""
    assert hasattr(BaseInfer, '__init__')
    assert inspect.ismethod(BaseInfer.__init__)
    assert inspect.signature(BaseInfer.__init__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='max_seq_len',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=bool,
            ),
            Parameter(
                name='kwargs',
                kind=Parameter.VAR_KEYWORD,
                annotation=Optional[Dict],
            ),
        ],
        return_annotation=Signature.empty,
    )


def test_abstract_method():
    r"""Ensure abstract method's signature."""
    assert 'gen' in BaseInfer.__abstractmethods__


def test_instance_attribute(
        max_seq_len: int,
        subclss_infer: BaseInfer,
):
    r"""Ensure instance attributes' signature."""
    assert subclss_infer.max_seq_len == max_seq_len


def test_static_method():
    r"""Ensure static methods' signature."""
    assert hasattr(BaseInfer, 'infer_parser')
    assert inspect.isfunction(BaseInfer.infer_parser)
    assert inspect.signature(BaseInfer.infer_parser) == Signature(
        parameters=[
            Parameter(
                name='parser',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=argparse.ArgumentParser,
            ),
        ],
        return_annotation=None,
    )
