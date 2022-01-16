"""Test :py:class:`lmp.model._res_lstm` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

from lmp.model._res_lstm import ResLSTMBlock, ResLSTMModel
from lmp.model._res_rnn import ResRNNBlock, ResRNNModel
from lmp.tknzr._base import BaseTknzr


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(ResLSTMModel)
  assert not inspect.isabstract(ResLSTMModel)
  assert issubclass(ResLSTMModel, ResRNNModel)
  assert inspect.isclass(ResLSTMBlock)
  assert not inspect.isabstract(ResLSTMBlock)
  assert issubclass(ResLSTMBlock, ResRNNBlock)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert isinstance(ResLSTMModel.model_name, str)
  assert ResLSTMModel.model_name == 'res-LSTM'
  assert ResLSTMModel.file_name == 'model-{}.pt'


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(ResLSTMBlock, '__init__')
  assert inspect.signature(ResLSTMBlock.__init__) == Signature(
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

  assert hasattr(ResLSTMModel, '__init__')
  assert inspect.signature(ResLSTMModel.__init__) == Signature(
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
  """Ensure inherent methods are same as base class."""
  assert (inspect.signature(ResRNNModel.forward) == inspect.signature(ResLSTMModel.forward))

  assert (inspect.signature(ResRNNModel.load) == inspect.signature(ResLSTMModel.load))

  assert (inspect.signature(ResRNNModel.loss_fn) == inspect.signature(ResLSTMModel.loss_fn))

  assert (inspect.signature(ResRNNModel.pred) == inspect.signature(ResLSTMModel.pred))

  assert (inspect.signature(ResRNNModel.ppl) == inspect.signature(ResLSTMModel.ppl))

  assert (inspect.signature(ResRNNModel.save) == inspect.signature(ResLSTMModel.save))

  assert (inspect.signature(ResRNNModel.train_parser) == inspect.signature(ResLSTMModel.train_parser))
