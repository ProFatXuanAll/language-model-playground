r"""Test :py:class:`lmp.dset.ChPoemDset` signature."""

import inspect
from typing import get_type_hints

from lmp.dset._base import BaseDset
from lmp.dset._ch_poem import ChPoemDset


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(ChPoemDset)
    assert not inspect.isabstract(ChPoemDset)
    assert issubclass(ChPoemDset, BaseDset)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert get_type_hints(ChPoemDset) == get_type_hints(BaseDset)
    assert ChPoemDset.df_ver == '唐'
    assert ChPoemDset.dset_name == 'chinese-poem'
    assert ChPoemDset.file_name == '{}.csv.zip'
    assert ChPoemDset.lang == 'zh'
    assert ChPoemDset.vers == [
        '元', '元末明初', '先秦', '南北朝', '唐', '唐末宋初', '宋', '宋末元初', '宋末金初', '明',
        '明末清初', '民國末當代初', '清', '清末民國初', '清末近現代初', '漢', '當代', '秦', '近現代',
        '近現代末當代初', '遼', '金', '金末元初', '隋', '隋末唐初', '魏晉', '魏晉末南北朝初',
    ]
    assert ChPoemDset.url == ''.join([
        'https://github.com/ProFatXuanAll',
        '/demo-dataset/raw/main/ch-poem',
    ])


def test_inherent_method():
    r'''Ensure inherent methods' signature are the same as base class.'''
    assert (
        inspect.signature(BaseDset.download)
        ==
        inspect.signature(ChPoemDset.download)
    )

    assert (
        inspect.signature(BaseDset.__getitem__)
        ==
        inspect.signature(ChPoemDset.__getitem__)
    )

    assert (
        inspect.signature(BaseDset.__init__)
        ==
        inspect.signature(ChPoemDset.__init__)
    )

    assert (
        inspect.signature(BaseDset.__iter__)
        ==
        inspect.signature(ChPoemDset.__iter__)
    )

    assert (
        inspect.signature(BaseDset.__len__)
        ==
        inspect.signature(ChPoemDset.__len__)
    )
