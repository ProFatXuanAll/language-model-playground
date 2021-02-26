r"""Test :py:class:`lmp.model._sattn_lstm` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

from lmp.model._base import BaseModel
from lmp.model._sattn_lstm import SAttnLSTMBlock, SAttnLSTMModel
from lmp.tknzr._base import BaseTknzr


def test_class():
    r"""Subclass only need to implement method __init__.
    """
    assert inspect.isclass(SAttnLSTMBlock)
    assert inspect.isclass(SAttnLSTMModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(SAttnLSTMModel.model_name, str)
    assert SAttnLSTMModel.model_name == 'sattn-LSTM'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(SAttnLSTMBlock, '__init__')
    assert inspect.signature(SAttnLSTMBlock.__init__) == Signature(
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

    assert hasattr(SAttnLSTMModel, '__init__')
    assert inspect.signature(SAttnLSTMModel.__init__) == Signature(
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
        inspect.signature(SAttnLSTMModel.forward)
    )

    assert (
        inspect.signature(BaseModel.loss_fn)
        ==
        inspect.signature(SAttnLSTMModel.loss_fn)
    )

    assert (
        inspect.signature(BaseModel.pred)
        ==
        inspect.signature(SAttnLSTMModel.pred)
    )

    assert (
        inspect.signature(BaseModel.ppl)
        ==
        inspect.signature(SAttnLSTMModel.ppl)
    )

    assert (
        inspect.signature(BaseModel.save)
        ==
        inspect.signature(SAttnLSTMModel.save)
    )

    assert (
        inspect.signature(BaseModel.load)
        ==
        inspect.signature(SAttnLSTMModel.load)
    )

    assert (
        inspect.signature(BaseModel.train_parser)
        ==
        inspect.signature(SAttnLSTMModel.train_parser)
    )
