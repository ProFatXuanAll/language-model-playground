r"""Test :py:class:`lmp.tknzr.WsTknzr` signature."""

import inspect

from lmp.tknzr._base import BaseTknzr
from lmp.tknzr._ws import WsTknzr


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(WsTknzr)
    assert not inspect.isabstract(WsTknzr)
    assert issubclass(WsTknzr, BaseTknzr)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert WsTknzr.bos_tk == BaseTknzr.bos_tk
    assert WsTknzr.bos_tkid == BaseTknzr.bos_tkid
    assert WsTknzr.eos_tk == BaseTknzr.eos_tk
    assert WsTknzr.eos_tkid == BaseTknzr.eos_tkid
    assert WsTknzr.file_name == BaseTknzr.file_name
    assert WsTknzr.pad_tk == BaseTknzr.pad_tk
    assert WsTknzr.pad_tkid == BaseTknzr.pad_tkid
    assert isinstance(WsTknzr.tknzr_name, str)
    assert WsTknzr.tknzr_name == 'whitespace'
    assert WsTknzr.unk_tk == BaseTknzr.unk_tk
    assert WsTknzr.unk_tkid == BaseTknzr.unk_tkid


def test_inherent_method():
    r'''Ensure inherent methods are same as baseclass.'''
    assert (
        inspect.signature(WsTknzr.__init__)
        ==
        inspect.signature(BaseTknzr.__init__)
    )
    assert WsTknzr.save == BaseTknzr.save
    assert (
        inspect.signature(WsTknzr.load)
        ==
        inspect.signature(BaseTknzr.load)
    )
    assert WsTknzr.norm == BaseTknzr.norm
    assert (
        inspect.signature(WsTknzr.tknz)
        ==
        inspect.signature(BaseTknzr.tknz)
    )
    assert (
        inspect.signature(WsTknzr.dtknz)
        ==
        inspect.signature(BaseTknzr.dtknz)
    )
    assert WsTknzr.enc == BaseTknzr.enc
    assert WsTknzr.dec == BaseTknzr.dec
    assert WsTknzr.batch_enc == BaseTknzr.batch_enc
    assert WsTknzr.batch_dec == BaseTknzr.batch_dec
    assert WsTknzr.build_vocab == BaseTknzr.build_vocab
    assert WsTknzr.train_parser == BaseTknzr.train_parser
