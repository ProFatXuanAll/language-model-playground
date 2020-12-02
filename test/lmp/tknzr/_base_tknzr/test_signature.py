r"""Test :py:class:`lmp.tknzr.BaseTknzr` signature."""

import inspect
from inspect import Parameter, Signature
from typing import ClassVar, Dict, Optional, Union, get_type_hints, Sequence, List

from lmp.tknzr._base import BaseTknzr


def test_class():
    r"""Ensure abstract class signature.

    Subclass only need to implement method tknzr and dtknzr.
    """
    assert inspect.isclass(BaseTknzr)
    assert inspect.isabstract(BaseTknzr)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert get_type_hints(BaseTknzr) == {
        'bos_tk': ClassVar[str],
        'bos_tkid': ClassVar[int],
        'eos_tk': ClassVar[str],
        'eos_tkid': ClassVar[int],
        'file_name': ClassVar[str],
        'pad_tk': ClassVar[str],
        'pad_tkid': ClassVar[int],
        'tknzr_name': ClassVar[str],
        'unk_tk': ClassVar[str],
        'unk_tkid': ClassVar[int],
    }
    assert BaseTknzr.bos_tk == '[bos]'
    assert BaseTknzr.bos_tkid == 0
    assert BaseTknzr.eos_tk == '[eos]'
    assert BaseTknzr.eos_tkid == 1
    assert BaseTknzr.file_name == 'tknzr.json'
    assert BaseTknzr.pad_tk == '[pad]'
    assert BaseTknzr.pad_tkid == 2
    assert BaseTknzr.tknzr_name == 'base'
    assert BaseTknzr.unk_tk == '[unk]'
    assert BaseTknzr.unk_tkid == 3


def test_class_method():
    r"""Ensure class methods' signature."""
    assert hasattr(BaseTknzr, 'load')
    assert inspect.ismethod(BaseTknzr.load)
    # TODO: add signature test for the function.
    assert inspect.signature(BaseTknzr.load) == Signature(
        parameters=[
            Parameter(
                name='cls',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='exp_name',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ]
    )


def test_instance_method(subclass_tknzr):
    r"""Ensure instance methods' signature."""
    assert hasattr(BaseTknzr, '__init__')
    assert inspect.ismethod(subclass_tknzr.__init__)
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
            Parameter(
                name='kwargs',
                kind=Parameter.VAR_KEYWORD,
                annotation=Optional[Dict],
            ),
        ],
        return_annotation=Signature.empty
    )

    # TODO: add signature test for the rest functions.
    assert hasattr(BaseTknzr, 'batch_enc')
    assert inspect.ismethod(subclass_tknzr.batch_enc)
    assert inspect.signature(BaseTknzr.batch_enc) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='batch_txt',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=Sequence[str],
            ),
            Parameter(
                name='max_seq_len',
                kind=Parameter.KEYWORD_ONLY,
                default=-1,
                annotation=int,
            ),
        ], 
        return_annotation=List[List[int]]
    )
    assert hasattr(BaseTknzr, 'batch_dec')
    assert inspect.ismethod(subclass_tknzr.batch_dec)
    assert inspect.signature(BaseTknzr.batch_dec) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='batch_tkids',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=Sequence[Sequence[int]],
            ),
            Parameter(
                name='rm_sp_tks',
                kind=Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool,
            ),
        ], 
        return_annotation=List[str]
    )

    assert hasattr(BaseTknzr, 'build_vocab')
    assert inspect.ismethod(subclass_tknzr.build_vocab)
    assert inspect.signature(BaseTknzr.build_vocab) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='batch_txt',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=Sequence[str],
            ),
        ], 
        return_annotation=Signature.empty
    )

    assert hasattr(BaseTknzr, 'dec')
    assert inspect.ismethod(subclass_tknzr.dec)
    assert inspect.signature(BaseTknzr.dec) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='tkids',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=Sequence[int],
            ),
            Parameter(
                name='rm_sp_tks',
                kind=Parameter.KEYWORD_ONLY,
                default=False,
                annotation=Optional[bool],
            ),
        ], 
        return_annotation=str
    )

    assert hasattr(BaseTknzr, 'dtknz')
    assert inspect.ismethod(subclass_tknzr.dtknz)
    assert inspect.signature(BaseTknzr.dtknz) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='tks',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=Sequence[str],
            ),
        ], 
        return_annotation=str
    )

    assert hasattr(BaseTknzr, 'enc')
    assert inspect.ismethod(subclass_tknzr.enc)
    assert inspect.signature(BaseTknzr.enc) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='txt',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=Parameter.empty,
            ),
            Parameter(
                name='max_seq_len',
                kind=Parameter.KEYWORD_ONLY,
                default=-1,
                annotation=Optional[int],
            ),
        ], 
        return_annotation=List[int]
    )

    assert hasattr(BaseTknzr, 'norm')
    assert inspect.ismethod(subclass_tknzr.norm)
    assert inspect.signature(BaseTknzr.norm) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='txt',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ], 
        return_annotation=str
    )

    assert hasattr(BaseTknzr, 'save')
    assert inspect.ismethod(subclass_tknzr.save)
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
    assert hasattr(BaseTknzr, 'tknz')
    assert inspect.ismethod(subclass_tknzr.tknz)
    assert inspect.signature(BaseTknzr.tknz) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='txt',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=str,
            ),
        ], 
        return_annotation=List[str]
    )


def test_abstract_method():
    r"""Ensure abstract method's signature."""
    assert 'dtknz' in BaseTknzr.__abstractmethods__
    assert 'tknz' in BaseTknzr.__abstractmethods__


def test_instance_attribute(
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
