r"""Test :py:class:`lmp.tknzr.BaseTknzr` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional, Union

from lmp.tknzr._base_tknzr import BaseTknzr


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert hasattr(BaseTknzr, 'bos_tk')
    assert isinstance(BaseTknzr.bos_tk, str)
    assert hasattr(BaseTknzr, 'bos_tkid')
    assert isinstance(BaseTknzr.bos_tkid, int)
    assert hasattr(BaseTknzr, 'eos_tk')
    assert isinstance(BaseTknzr.eos_tk, str)
    assert hasattr(BaseTknzr, 'eos_tkid')
    assert isinstance(BaseTknzr.eos_tkid, int)
    assert hasattr(BaseTknzr, 'file_name')
    assert isinstance(BaseTknzr.file_name, str)
    assert hasattr(BaseTknzr, 'pad_tk')
    assert isinstance(BaseTknzr.pad_tk, str)
    assert hasattr(BaseTknzr, 'pad_tkid')
    assert isinstance(BaseTknzr.pad_tkid, int)
    assert hasattr(BaseTknzr, 'tknzr_name')
    assert isinstance(BaseTknzr.tknzr_name, str)
    assert hasattr(BaseTknzr, 'unk_tk')
    assert isinstance(BaseTknzr.unk_tk, str)
    assert hasattr(BaseTknzr, 'unk_tkid')
    assert isinstance(BaseTknzr.unk_tkid, int)


def test_class_method():
    r"""Ensure class methods' signature."""
    assert hasattr(BaseTknzr, 'load')


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(BaseTknzr, '__init__')
    assert inspect.signature(BaseTknzr.__init__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='is_uncased',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=bool,
            ),
            Parameter(
                name='max_vocab',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=int,
            ),
            Parameter(
                name='min_count',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=int,
            ),
            Parameter(
                name='tk2id',
                kind=Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[Dict[str, int]],
            ),
        ],
        return_annotation=Signature.empty
    )

    assert hasattr(BaseTknzr, 'save')
    assert inspect.signature(BaseTknzr.save) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='exp_name',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ],
        return_annotation=None
    )
    # TODO: add signature test for the rest function.
    assert hasattr(BaseTknzr, 'batch_enc')
    assert hasattr(BaseTknzr, 'batch_dec')
    assert hasattr(BaseTknzr, 'build_vocab')
    assert hasattr(BaseTknzr, 'dec')
    assert hasattr(BaseTknzr, 'dtknz')
    assert hasattr(BaseTknzr, 'enc')
    assert hasattr(BaseTknzr, 'norm')
    assert hasattr(BaseTknzr, 'pad_to_max')
    assert hasattr(BaseTknzr, 'save')
    assert hasattr(BaseTknzr, 'tknz')
    assert hasattr(BaseTknzr, 'trunc_to_max')


def test_abstract_class(subclass_tknzr: BaseTknzr):
    r"""Subclass only need to implement method tknzr and dtknzr."""
    assert subclass_tknzr


def test_inst(
        is_uncased: bool,
        max_vocab: int,
        min_count: int,
        subclass_tknzr: BaseTknzr,
        tk2id: Union[None, Dict[str, int]],
):
    r"""Ensure instance attributes' signature."""
    assert subclass_tknzr.is_uncased == is_uncased
    assert subclass_tknzr.max_vocab == max_vocab
    assert subclass_tknzr.min_count == min_count
    if tk2id is not None:
        assert subclass_tknzr.tk2id == tk2id
        assert subclass_tknzr.id2tk == {v: k for k, v in tk2id.items()}
    else:
        assert subclass_tknzr.id2tk == {
            BaseTknzr.bos_tkid: BaseTknzr.bos_tk,
            BaseTknzr.eos_tkid: BaseTknzr.eos_tk,
            BaseTknzr.pad_tkid: BaseTknzr.pad_tk,
            BaseTknzr.unk_tkid: BaseTknzr.unk_tk,
        }
        assert subclass_tknzr.tk2id == {
            BaseTknzr.bos_tk: BaseTknzr.bos_tkid,
            BaseTknzr.eos_tk: BaseTknzr.eos_tkid,
            BaseTknzr.pad_tk: BaseTknzr.pad_tkid,
            BaseTknzr.unk_tk: BaseTknzr.unk_tkid,
        }
