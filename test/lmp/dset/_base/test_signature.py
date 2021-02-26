r"""Test :py:class:`lmp.dset.BaseDset` signature."""

import inspect
from inspect import Parameter, Signature
from typing import ClassVar, Iterator, List, Optional, get_type_hints

from lmp.dset._base import BaseDset


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(BaseDset)
    assert not inspect.isabstract(BaseDset)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert get_type_hints(BaseDset) == {
        'df_ver': ClassVar[str],
        'dset_name': ClassVar[str],
        'file_name': ClassVar[str],
        'lang': ClassVar[str],
        'vers': ClassVar[List[str]],
        'url': ClassVar[str],
    }
    assert BaseDset.df_ver == ''
    assert BaseDset.dset_name == 'base'
    assert BaseDset.file_name == ''
    assert BaseDset.lang == ''
    assert BaseDset.vers == []
    assert BaseDset.url == ''


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(BaseDset, '__init__')
    assert inspect.signature(BaseDset.__init__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='ver',
                kind=Parameter.KEYWORD_ONLY,
                annotation=Optional[str],
                default=None,
            ),
        ],
        return_annotation=Signature.empty,
    )
    assert hasattr(BaseDset, '__iter__')
    assert inspect.signature(BaseDset.__iter__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
        ],
        return_annotation=Iterator[str],
    )
    assert hasattr(BaseDset, '__len__')
    assert inspect.signature(BaseDset.__len__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
        ],
        return_annotation=int,
    )
    assert hasattr(BaseDset, '__getitem__')
    assert inspect.signature(BaseDset.__getitem__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='idx',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=int,
            )
        ],
        return_annotation=str,
    )
    assert hasattr(BaseDset, 'download')
    assert inspect.signature(BaseDset.download) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
        ],
        return_annotation=None,
    )
