r"""Test :py:class:`lmp.model._sattn_gru` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

from lmp.model._base import BaseModel
from lmp.model._sattn_gru import SAttnGRUBlock, SAttnGRUModel
from lmp.tknzr._base import BaseTknzr


def test_class():
    r"""Subclass only need to implement method __init__.
    """
    assert inspect.isclass(SAttnGRUBlock)
    assert inspect.isclass(SAttnGRUModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(SAttnGRUModel.model_name, str)
    assert SAttnGRUModel.model_name == 'sattn-GRU'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(SAttnGRUBlock, '__init__')
    assert inspect.signature(SAttnGRUBlock.__init__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='d_hid',
                kind=Parameter.KEYWORD_ONLY,
                annotation=int,
                default=Parameter.empty,
            ),
            Parameter(
                name='n_hid_lyr',
                kind=Parameter.KEYWORD_ONLY,
                annotation=int,
                default=Parameter.empty,
            ),
            Parameter(
                name='p_hid',
                kind=Parameter.KEYWORD_ONLY,
                annotation=float,
                default=Parameter.empty,
            ),
            Parameter(
                name='kwargs',
                kind=Parameter.VAR_KEYWORD,
                annotation=Optional[Dict],
            ),
        ],
    )

    assert hasattr(SAttnGRUModel, '__init__')
    assert inspect.signature(SAttnGRUModel.__init__) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='d_emb',
                kind=Parameter.KEYWORD_ONLY,
                annotation=int,
                default=Parameter.empty,
            ),
            Parameter(
                name='d_hid',
                kind=Parameter.KEYWORD_ONLY,
                annotation=int,
                default=Parameter.empty,
            ),
            Parameter(
                name='n_hid_lyr',
                kind=Parameter.KEYWORD_ONLY,
                annotation=int,
                default=Parameter.empty,
            ),
            Parameter(
                name='n_post_hid_lyr',
                kind=Parameter.KEYWORD_ONLY,
                annotation=int,
                default=Parameter.empty,
            ),
            Parameter(
                name='n_pre_hid_lyr',
                kind=Parameter.KEYWORD_ONLY,
                annotation=int,
                default=Parameter.empty,
            ),
            Parameter(
                name='p_emb',
                kind=Parameter.KEYWORD_ONLY,
                annotation=float,
                default=Parameter.empty,
            ),
            Parameter(
                name='p_hid',
                kind=Parameter.KEYWORD_ONLY,
                annotation=float,
                default=Parameter.empty,
            ),
            Parameter(
                name='tknzr',
                kind=Parameter.KEYWORD_ONLY,
                annotation=BaseTknzr,
                default=Parameter.empty,
            ),
            Parameter(
                name='kwargs',
                kind=Parameter.VAR_KEYWORD,
                annotation=Optional[Dict],
            ),
        ],
    )


def test_inherent_method():
    r'''Ensure inherent methods' signature are same as base class.'''
    assert (
        inspect.signature(BaseModel.forward)
        ==
        inspect.signature(SAttnGRUModel.forward)
    )
    assert (
        inspect.signature(BaseModel.loss_fn)
        ==
        inspect.signature(SAttnGRUModel.loss_fn)
    )

    assert (
        inspect.signature(BaseModel.pred)
        ==
        inspect.signature(SAttnGRUModel.pred)
    )

    assert (
        inspect.signature(BaseModel.ppl)
        ==
        inspect.signature(SAttnGRUModel.ppl)
    )

    assert (
        inspect.signature(BaseModel.save)
        ==
        inspect.signature(SAttnGRUModel.save)
    )

    assert (
        inspect.signature(BaseModel.load)
        ==
        inspect.signature(SAttnGRUModel.load)
    )

    assert (
        inspect.signature(BaseModel.train_parser)
        ==
        inspect.signature(SAttnGRUModel.train_parser)
    )
