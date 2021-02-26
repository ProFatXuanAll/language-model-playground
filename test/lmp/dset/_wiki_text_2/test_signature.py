r"""Test :py:class:`lmp.dset.WikiText2Dset` signature."""

import inspect
from typing import get_type_hints

from lmp.dset._base import BaseDset
from lmp.dset._wiki_text_2 import WikiText2Dset


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(WikiText2Dset)
    assert not inspect.isabstract(WikiText2Dset)
    assert issubclass(WikiText2Dset, BaseDset)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert get_type_hints(WikiText2Dset) == get_type_hints(BaseDset)
    assert WikiText2Dset.df_ver == 'train'
    assert WikiText2Dset.dset_name == 'wikitext-2'
    assert WikiText2Dset.file_name == 'wiki.{}.tokens.zip'
    assert WikiText2Dset.lang == 'en'
    assert WikiText2Dset.vers == ['test', 'train', 'valid']
    assert WikiText2Dset.url == ''.join([
        'https://github.com/ProFatXuanAll',
        '/demo-dataset/raw/main/wikitext-2',
    ])


def test_inherent_method():
    r'''Ensure inherent methods' signature are the same as base class.'''
    assert (
        inspect.signature(BaseDset.__getitem__)
        ==
        inspect.signature(WikiText2Dset.__getitem__)
    )

    assert (
        inspect.signature(BaseDset.__init__)
        ==
        inspect.signature(WikiText2Dset.__init__)
    )

    assert (
        inspect.signature(BaseDset.__iter__)
        ==
        inspect.signature(WikiText2Dset.__iter__)
    )

    assert (
        inspect.signature(BaseDset.__len__)
        ==
        inspect.signature(WikiText2Dset.__len__)
    )

    assert (
        inspect.signature(BaseDset.download)
        ==
        inspect.signature(WikiText2Dset.download)
    )
