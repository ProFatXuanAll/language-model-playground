r"""Test :py:class:`lmp.model._res_sattn_gru` signature."""

import inspect
from inspect import Parameter, Signature
from typing import (Optional, Dict)

from lmp.model._res_sattn_gru import ResSAttnGRUBlock, ResSAttnGRUModel
from lmp.tknzr._base import BaseTknzr
from lmp.model._base import BaseModel


def test_class():
    r"""Subclass only need to implement method __init__.
    """
    assert inspect.isclass(ResSAttnGRUBlock)
    assert inspect.isclass(ResSAttnGRUModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(ResSAttnGRUModel.model_name, str)
    assert ResSAttnGRUModel.model_name == 'res-sattn-GRU'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(ResSAttnGRUBlock, '__init__')
    assert inspect.signature(ResSAttnGRUBlock.__init__) == Signature(
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
        return_annotation=Signature.empty,
    )

    assert hasattr(ResSAttnGRUModel, '__init__')
    assert inspect.signature(ResSAttnGRUModel.__init__) == Signature(
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
        return_annotation=Signature.empty,
    )


def test_inherent_method():
    r'''Ensure inherent methods are same as baseclass.'''
    assert inspect.signature(
        BaseModel.forward) == inspect.signature(
        ResSAttnGRUModel.forward)

    assert inspect.signature(BaseModel.loss_fn) == inspect.signature(
        ResSAttnGRUModel.loss_fn)

    assert inspect.signature(BaseModel.pred) == inspect.signature(
        ResSAttnGRUModel.pred)

    assert inspect.signature(BaseModel.ppl) == inspect.signature(
        ResSAttnGRUModel.ppl)

    assert inspect.signature(BaseModel.save) == inspect.signature(
        ResSAttnGRUModel.save)

    assert inspect.signature(BaseModel.load) == inspect.signature(
        ResSAttnGRUModel.load)

    assert inspect.signature(BaseModel.train_parser) == inspect.signature(
        ResSAttnGRUModel.train_parser)
