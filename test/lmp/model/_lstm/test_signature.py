r"""Test :py:class:`lmp.model._lstm` signature."""

import inspect
from inspect import Parameter, Signature
from typing import (Optional, Dict)

from lmp.model._lstm import LSTMModel
from lmp.tknzr._base import BaseTknzr
from lmp.model._base import BaseModel


def test_class():
    r"""Subclass only need to implement method __init__.
    """
    assert inspect.isclass(LSTMModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(LSTMModel.model_name, str)
    assert LSTMModel.model_name == 'LSTM'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(LSTMModel, '__init__')
    assert inspect.signature(LSTMModel.__init__) == Signature(
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
        LSTMModel.forward)

    assert inspect.signature(BaseModel.loss_fn) == inspect.signature(
        LSTMModel.loss_fn)

    assert inspect.signature(BaseModel.pred) == inspect.signature(
        LSTMModel.pred)

    assert inspect.signature(BaseModel.ppl) == inspect.signature(
        LSTMModel.ppl)

    assert inspect.signature(BaseModel.save) == inspect.signature(
        LSTMModel.save)

    assert inspect.signature(BaseModel.load) == inspect.signature(
        LSTMModel.load)

    assert inspect.signature(BaseModel.train_parser) == inspect.signature(
        LSTMModel.train_parser)
