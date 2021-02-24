r"""Test :py:class:`lmp.model._sattn_rnn` signature."""

import inspect
from inspect import Parameter, Signature
from typing import (Optional, Dict)
import torch

from lmp.model._sattn_rnn import SAttnRNNBlock, SAttnRNNModel
from lmp.tknzr._base import BaseTknzr
from lmp.model._base import BaseModel


def test_class():
    r"""Subclass only need to implement method __init__.
    """
    assert inspect.isclass(SAttnRNNBlock)
    assert inspect.isclass(SAttnRNNModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(SAttnRNNModel.model_name, str)
    assert SAttnRNNModel.model_name == 'sattn-RNN'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(SAttnRNNBlock, '__init__')
    assert inspect.signature(SAttnRNNBlock.__init__) == Signature(
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

    assert hasattr(SAttnRNNBlock, 'forward')
    assert inspect.signature(SAttnRNNBlock.forward) == Signature(
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

    assert hasattr(SAttnRNNModel, '__init__')
    assert inspect.signature(SAttnRNNModel.__init__) == Signature(
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

    assert hasattr(SAttnRNNModel, 'create_mask')
    assert inspect.signature(SAttnRNNModel.create_mask) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='batch_prev_tkids',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
                default=Parameter.empty,
            ),
        ],
        return_annotation=torch.Tensor,
    )

    assert hasattr(SAttnRNNModel, 'forward')
    assert inspect.signature(SAttnRNNModel.forward) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='batch_prev_tkids',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
                default=Parameter.empty,
            ),
        ],
        return_annotation=torch.Tensor,
    )


def test_inherent_method():
    r'''Ensure inherent methods are same as baseclass.'''
    assert inspect.signature(
        BaseModel.forward) == inspect.signature(
        SAttnRNNModel.forward)

    assert inspect.signature(BaseModel.loss_fn) == inspect.signature(
        SAttnRNNModel.loss_fn)

    assert inspect.signature(BaseModel.pred) == inspect.signature(
        SAttnRNNModel.pred)

    assert inspect.signature(BaseModel.ppl) == inspect.signature(
        SAttnRNNModel.ppl)

    assert inspect.signature(BaseModel.save) == inspect.signature(
        SAttnRNNModel.save)

    assert inspect.signature(BaseModel.load) == inspect.signature(
        SAttnRNNModel.load)

    assert inspect.signature(BaseModel.train_parser) == inspect.signature(
        SAttnRNNModel.train_parser)
