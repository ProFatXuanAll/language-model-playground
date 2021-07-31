r"""Test :py:class:`lmp.model._res_rnn` signature."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

import torch

from lmp.model._base import BaseModel
from lmp.model._res_rnn import ResRNNBlock, ResRNNModel
from lmp.model._rnn import RNNModel
from lmp.tknzr._base import BaseTknzr


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(ResRNNModel)
    assert not inspect.isabstract(ResRNNModel)
    assert issubclass(ResRNNModel, RNNModel)
    assert inspect.isclass(ResRNNBlock)
    assert not inspect.isabstract(ResRNNBlock)
    assert issubclass(ResRNNBlock, torch.nn.Module)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(ResRNNModel.model_name, str)
    assert ResRNNModel.model_name == 'res-RNN'
    assert ResRNNModel.file_name == 'model-{}.pt'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(ResRNNBlock, '__init__')
    assert inspect.signature(ResRNNBlock.__init__) == Signature(
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

    assert hasattr(ResRNNBlock, 'forward')
    assert inspect.signature(ResRNNBlock.forward) == Signature(
        parameters=[
            Parameter(
                name='self',
                kind=Parameter.POSITIONAL_OR_KEYWORD,
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

    assert hasattr(ResRNNModel, '__init__')
    assert inspect.signature(ResRNNModel.__init__) == Signature(
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
        inspect.signature(ResRNNModel.forward)
    )

    assert (
        inspect.signature(BaseModel.load)
        ==
        inspect.signature(ResRNNModel.load)
    )

    assert (
        inspect.signature(BaseModel.loss_fn)
        ==
        inspect.signature(ResRNNModel.loss_fn)
    )

    assert (
        inspect.signature(BaseModel.pred)
        ==
        inspect.signature(ResRNNModel.pred)
    )

    assert (
        inspect.signature(BaseModel.ppl)
        ==
        inspect.signature(ResRNNModel.ppl)
    )

    assert (
        inspect.signature(BaseModel.save)
        ==
        inspect.signature(ResRNNModel.save)
    )

    assert (
        inspect.signature(BaseModel.train_parser)
        ==
        inspect.signature(ResRNNModel.train_parser)
    )
