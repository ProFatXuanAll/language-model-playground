r"""Test :py:class:`lmp.tknzr.CharTknzr` signature."""

import inspect

from lmp.tknzr import BaseTknzr, CharTknzr


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(CharTknzr)
    assert not inspect.isabstract(CharTknzr)
    assert issubclass(CharTknzr, BaseTknzr)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert CharTknzr.bos_tk == BaseTknzr.bos_tk
    assert CharTknzr.bos_tkid == BaseTknzr.bos_tkid
    assert CharTknzr.eos_tk == BaseTknzr.eos_tk
    assert CharTknzr.eos_tkid == BaseTknzr.eos_tkid
    assert CharTknzr.file_name == BaseTknzr.file_name
    assert CharTknzr.pad_tk == BaseTknzr.pad_tk
    assert CharTknzr.pad_tkid == BaseTknzr.pad_tkid
    assert isinstance(CharTknzr.tknzr_name, str)
    assert CharTknzr.tknzr_name == 'character'
    assert CharTknzr.unk_tk == BaseTknzr.unk_tk
    assert CharTknzr.unk_tkid == BaseTknzr.unk_tkid


def test_inherent_method():
    r'''Ensure inherent methods are same as baseclass.'''
    assert (
        inspect.signature(CharTknzr.__init__)
        ==
        inspect.signature(BaseTknzr.__init__)
    )
    assert CharTknzr.save == BaseTknzr.save
    assert (
        inspect.signature(CharTknzr.load)
        ==
        inspect.signature(BaseTknzr.load)
    )
    assert CharTknzr.norm == BaseTknzr.norm
    assert (
        inspect.signature(CharTknzr.tknz)
        ==
        inspect.signature(BaseTknzr.tknz)
    )
    assert (
        inspect.signature(CharTknzr.dtknz)
        ==
        inspect.signature(BaseTknzr.dtknz)
    )
    assert CharTknzr.enc == BaseTknzr.enc
    assert CharTknzr.dec == BaseTknzr.dec
    assert CharTknzr.batch_enc == BaseTknzr.batch_enc
    assert CharTknzr.batch_dec == BaseTknzr.batch_dec
    assert CharTknzr.build_vocab == BaseTknzr.build_vocab
    assert CharTknzr.train_parser == BaseTknzr.train_parser
