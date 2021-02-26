r"""Test :py:class:`lmp.model._rnn` signature."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

import torch

from lmp.model._base import BaseModel
from lmp.model._rnn import RNNModel
from lmp.tknzr._base import BaseTknzr


def test_class():
    r"""Subclass only need to implement method __init__.
    """
    assert inspect.isclass(RNNModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(RNNModel.model_name, str)
    assert RNNModel.model_name == 'RNN'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(RNNModel, '__init__')
    assert inspect.signature(RNNModel.__init__) == Signature(
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

    assert hasattr(RNNModel, 'forward')
    assert inspect.signature(RNNModel.forward) == Signature(
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

    assert hasattr(RNNModel, 'loss_fn')
    assert inspect.signature(RNNModel.loss_fn) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
            ),
            Parameter(
                name='batch_next_tkids',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=torch.Tensor,
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

    assert hasattr(RNNModel, 'pred')
    assert inspect.signature(RNNModel.pred) == Signature(
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


def test_static_method():
    r"""Ensure static methods' signature."""
    assert hasattr(RNNModel, 'train_parser')
    assert inspect.isfunction(RNNModel.train_parser)
    assert inspect.signature(RNNModel.train_parser) == Signature(
        parameters=[
            Parameter(
                name='parser',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty,
                annotation=argparse.ArgumentParser,
            ),
        ],
        return_annotation=None,
    )


def test_inherent_method():
    r'''Ensure inherent methods' signature are same as base class.'''
    assert (
        inspect.signature(BaseModel.forward)
        ==
        inspect.signature(RNNModel.forward)
    )

    assert (
        inspect.signature(BaseModel.loss_fn)
        ==
        inspect.signature(RNNModel.loss_fn)
    )

    assert (
        inspect.signature(BaseModel.pred)
        ==
        inspect.signature(RNNModel.pred)
    )

    assert (
        inspect.signature(BaseModel.ppl)
        ==
        inspect.signature(RNNModel.ppl)
    )

    assert (
        inspect.signature(BaseModel.save)
        ==
        inspect.signature(RNNModel.save)
    )

    assert (
        inspect.signature(BaseModel.load)
        ==
        inspect.signature(RNNModel.load)
    )

    assert (
        inspect.signature(BaseModel.train_parser)
        ==
        inspect.signature(RNNModel.train_parser)
    )
