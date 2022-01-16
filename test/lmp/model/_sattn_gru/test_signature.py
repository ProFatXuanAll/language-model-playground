"""Test :py:class:`lmp.model._sattn_gru` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

from lmp.model._sattn_gru import (SAttnGRUBlock, SAttnGRUModel, SAttnRNNBlock,
                                  SAttnRNNModel)
from lmp.tknzr._base import BaseTknzr


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(SAttnGRUBlock)
  assert not inspect.isabstract(SAttnGRUBlock)
  assert issubclass(SAttnGRUBlock, SAttnRNNBlock)
  assert inspect.isclass(SAttnGRUModel)
  assert not inspect.isabstract(SAttnGRUModel)
  assert issubclass(SAttnGRUModel, SAttnRNNModel)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert isinstance(SAttnGRUModel.model_name, str)
  assert SAttnGRUModel.model_name == 'sattn-GRU'
  assert SAttnGRUModel.file_name == 'model-{}.pt'


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
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
  """Ensure inherent methods are same as base class."""
  assert (inspect.signature(SAttnRNNModel.forward) == inspect.signature(SAttnGRUModel.forward))

  assert (inspect.signature(SAttnRNNModel.load) == inspect.signature(SAttnGRUModel.load))

  assert (inspect.signature(SAttnRNNModel.loss_fn) == inspect.signature(SAttnGRUModel.loss_fn))

  assert (inspect.signature(SAttnRNNModel.pred) == inspect.signature(SAttnGRUModel.pred))

  assert (inspect.signature(SAttnRNNModel.ppl) == inspect.signature(SAttnGRUModel.ppl))

  assert (inspect.signature(SAttnRNNModel.save) == inspect.signature(SAttnGRUModel.save))

  assert (inspect.signature(SAttnRNNModel.train_parser) == inspect.signature(SAttnGRUModel.train_parser))
