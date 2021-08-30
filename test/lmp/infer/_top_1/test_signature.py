r"""Test :py:class:`lmp.infer.Top1Infer` signature."""

import inspect
from typing import get_type_hints

from lmp.infer import BaseInfer, Top1Infer


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(Top1Infer)
    assert not inspect.isabstract(Top1Infer)
    assert issubclass(Top1Infer, BaseInfer)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    print(get_type_hints(Top1Infer))
    assert get_type_hints(Top1Infer) == get_type_hints(BaseInfer)
    assert Top1Infer.hard_max_seq_len == BaseInfer.hard_max_seq_len
    assert Top1Infer.infer_name == 'top-1'


def test_inherent_method():
    r'''Ensure inherent methods' signature are the same as base class.'''
    assert (
        inspect.signature(Top1Infer.__init__)
        ==
        inspect.signature(BaseInfer.__init__)
    )
    assert (
        inspect.signature(Top1Infer.gen)
        ==
        inspect.signature(BaseInfer.gen)
    )
    assert Top1Infer.infer_parser == BaseInfer.infer_parser
