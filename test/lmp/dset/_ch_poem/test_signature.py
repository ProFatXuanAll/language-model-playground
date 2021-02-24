r"""Test :py:class:`lmp.dset.ChPoemDset` signature."""

import inspect
from inspect import Parameter, Signature
from typing import (ClassVar, List, Optional,
                    get_type_hints)

from lmp.dset._base import BaseDset
from lmp.dset._ch_poem import ChPoemDset


def test_class():
    r"""Ensure abstract class signature.

    Subclass only need to implement method __init__.
    """
    assert inspect.isclass(ChPoemDset)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert get_type_hints(ChPoemDset) == {
        'df_ver': ClassVar[str],
        'dset_name': ClassVar[str],
        'file_name': ClassVar[str],
        'lang': ClassVar[str],
        'vers': ClassVar[List[str]],
        'url': ClassVar[str],
    }
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


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(ChPoemDset, '__init__')
    assert inspect.signature(ChPoemDset.__init__) == Signature(
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


def test_inherent_method():
    r'''Ensure inherent methods are same as baseclass.'''
    assert inspect.signature(
        BaseDset.__init__) == inspect.signature(
        ChPoemDset.__init__)

    assert inspect.signature(
        BaseDset.__iter__) == inspect.signature(
        ChPoemDset.__iter__)

    assert inspect.signature(
        BaseDset.__len__) == inspect.signature(
        ChPoemDset.__len__)

    assert inspect.signature(
        BaseDset.__getitem__) == inspect.signature(
        ChPoemDset.__getitem__)

    assert inspect.signature(
        BaseDset.download) == inspect.signature(
        ChPoemDset.download)
