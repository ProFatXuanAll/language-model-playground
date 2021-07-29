r"""Test :py:class:`lmp.model._res_sattn_rnn` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

import torch

from lmp.model._base import BaseModel
from lmp.model._res_sattn_rnn import ResSAttnRNNBlock, ResSAttnRNNModel
from lmp.model._sattn_rnn import SAttnRNNBlock, SAttnRNNModel
from lmp.tknzr._base import BaseTknzr


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(ResSAttnRNNBlock)
    assert not inspect.isabstract(ResSAttnRNNBlock)
    assert issubclass(ResSAttnRNNBlock, SAttnRNNBlock)
    assert inspect.isclass(ResSAttnRNNModel)
    assert not inspect.isabstract(ResSAttnRNNModel)
    assert issubclass(ResSAttnRNNModel, SAttnRNNModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(ResSAttnRNNModel.model_name, str)
    assert ResSAttnRNNModel.model_name == 'res-sattn-RNN'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(ResSAttnRNNBlock, 'forward')
    assert inspect.signature(ResSAttnRNNBlock.forward) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='batch_tk_mask',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
                default=Parameter.empty,
            ),
            Parameter(
                name='batch_tk_reps',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
                default=Parameter.empty,
            ),
        ],
        return_annotation=torch.Tensor,
    )

    assert hasattr(ResSAttnRNNModel, '__init__')
    assert inspect.signature(ResSAttnRNNModel.__init__) == Signature(
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
        inspect.signature(ResSAttnRNNModel.forward)
    )
    assert (
        inspect.signature(BaseModel.loss_fn)
        ==
        inspect.signature(ResSAttnRNNModel.loss_fn)
    )

    assert (
        inspect.signature(BaseModel.pred)
        ==
        inspect.signature(ResSAttnRNNModel.pred)
    )

    assert (
        inspect.signature(BaseModel.ppl)
        ==
        inspect.signature(ResSAttnRNNModel.ppl)
    )

    assert (
        inspect.signature(BaseModel.save)
        ==
        inspect.signature(ResSAttnRNNModel.save)
    )
    assert (
        inspect.signature(BaseModel.load)
        ==
        inspect.signature(ResSAttnRNNModel.load)
    )

    assert (
        inspect.signature(BaseModel.train_parser)
        ==
        inspect.signature(ResSAttnRNNModel.train_parser)
    )
