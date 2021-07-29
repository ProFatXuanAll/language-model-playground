r"""Test :py:class:`lmp.model._gru` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

from lmp.model._base import BaseModel
from lmp.model._gru import GRUModel
from lmp.tknzr._base import BaseTknzr


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(GRUModel)
    assert not inspect.isabstract(GRUModel)
    assert issubclass(GRUModel, BaseModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(GRUModel.model_name, str)
    assert GRUModel.model_name == 'GRU'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(GRUModel, '__init__')
    assert inspect.signature(GRUModel.__init__) == Signature(
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
    r'''Ensure inherent methods' signature are same as base class.'''
    assert (
        inspect.signature(BaseModel.forward)
        ==
        inspect.signature(GRUModel.forward)
    )

    assert (
        inspect.signature(BaseModel.loss_fn)
        ==
        inspect.signature(GRUModel.loss_fn)
    )

    assert (
        inspect.signature(BaseModel.pred)
        ==
        inspect.signature(GRUModel.pred)
    )

    assert (
        inspect.signature(BaseModel.ppl)
        ==
        inspect.signature(GRUModel.ppl)
    )

    assert (
        inspect.signature(BaseModel.save)
        ==
        inspect.signature(GRUModel.save)
    )

    assert (
        inspect.signature(BaseModel.load)
        ==
        inspect.signature(GRUModel.load)
    )

    assert (
        inspect.signature(BaseModel.train_parser)
        == inspect.signature(GRUModel.train_parser)
    )
