r"""Test :py:class:`lmp.dset.WikiText2Dset` signature."""

import inspect
from inspect import Parameter, Signature
from typing import (ClassVar, List, Optional,
                    get_type_hints)

from lmp.dset._base import BaseDset
from lmp.dset._wiki_text_2 import WikiText2Dset


def test_class():
    r"""Ensure abstract class signature.

    Subclass only need to implement method __init__.
    """
    assert inspect.isclass(WikiText2Dset)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert get_type_hints(WikiText2Dset) == {
        'df_ver': ClassVar[str],
        'dset_name': ClassVar[str],
        'file_name': ClassVar[str],
        'lang': ClassVar[str],
        'vers': ClassVar[List[str]],
        'url': ClassVar[str],
    }
    assert WikiText2Dset.df_ver == 'train'
    assert WikiText2Dset.dset_name == 'wikitext-2'
    assert WikiText2Dset.file_name == 'wiki.{}.tokens.zip'
    assert WikiText2Dset.lang == 'en'
    assert WikiText2Dset.vers == ['test', 'train', 'valid']
    assert WikiText2Dset.url == ''.join([
        'https://github.com/ProFatXuanAll',
        '/demo-dataset/raw/main/wikitext-2',
    ])


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(WikiText2Dset, '__init__')
    assert inspect.signature(WikiText2Dset.__init__) == Signature(
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


def test_subclass_method():
    r'''Ensure inherent methods are same as baseclass.'''
    assert inspect.signature(
        BaseDset.__init__) == inspect.signature(
        WikiText2Dset.__init__)

    assert inspect.signature(
        BaseDset.__iter__) == inspect.signature(
        WikiText2Dset.__iter__)

    assert inspect.signature(
        BaseDset.__len__) == inspect.signature(
        WikiText2Dset.__len__)

    assert inspect.signature(
        BaseDset.__getitem__) == inspect.signature(
        WikiText2Dset.__getitem__)

    assert inspect.signature(
        BaseDset.download) == inspect.signature(
        WikiText2Dset.download)
