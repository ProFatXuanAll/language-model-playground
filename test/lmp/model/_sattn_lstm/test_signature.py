"""Test :py:class:`lmp.model._sattn_lstm` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

from lmp.model._sattn_lstm import SAttnLSTMBlock, SAttnLSTMModel
from lmp.model._sattn_rnn import SAttnRNNBlock, SAttnRNNModel
from lmp.tknzr._base import BaseTknzr


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(SAttnLSTMBlock)
  assert not inspect.isabstract(SAttnLSTMBlock)
  assert issubclass(SAttnLSTMBlock, SAttnRNNBlock)
  assert inspect.isclass(SAttnLSTMModel)
  assert not inspect.isabstract(SAttnLSTMModel)
  assert issubclass(SAttnLSTMModel, SAttnRNNModel)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert isinstance(SAttnLSTMModel.model_name, str)
  assert SAttnLSTMModel.model_name == 'sattn-LSTM'
  assert SAttnLSTMModel.file_name == 'model-{}.pt'


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
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
  """Ensure inherent methods are same as base class."""
  assert (inspect.signature(SAttnRNNModel.forward) == inspect.signature(SAttnLSTMModel.forward))

  assert (inspect.signature(SAttnRNNModel.load) == inspect.signature(SAttnLSTMModel.load))

  assert (inspect.signature(SAttnRNNModel.loss_fn) == inspect.signature(SAttnLSTMModel.loss_fn))

  assert (inspect.signature(SAttnRNNModel.pred) == inspect.signature(SAttnLSTMModel.pred))

  assert (inspect.signature(SAttnRNNModel.ppl) == inspect.signature(SAttnLSTMModel.ppl))

  assert (inspect.signature(SAttnRNNModel.save) == inspect.signature(SAttnLSTMModel.save))

  assert (inspect.signature(SAttnRNNModel.train_parser) == inspect.signature(SAttnLSTMModel.train_parser))
