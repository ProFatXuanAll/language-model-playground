"""Test :py:class:`lmp.model` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

from lmp.model import BaseModel, RNNModel
from lmp.tknzr import BaseTknzr


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(RNNModel)
  assert not inspect.isabstract(RNNModel)
  assert issubclass(RNNModel, BaseModel)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert isinstance(RNNModel.model_name, str)
  assert RNNModel.model_name == 'RNN'
  assert RNNModel.file_name == BaseModel.file_name


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
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


def test_inherent_method():
  """Ensure inherent methods are same as base class."""
  assert (inspect.signature(RNNModel.forward) == inspect.signature(BaseModel.forward))
  assert (inspect.signature(RNNModel.load) == inspect.signature(BaseModel.load))
  assert (inspect.signature(RNNModel.loss_fn) == inspect.signature(BaseModel.loss_fn))
  assert (inspect.signature(RNNModel.pred) == inspect.signature(BaseModel.pred))
  assert (inspect.signature(RNNModel.ppl) == inspect.signature(BaseModel.ppl))
  assert RNNModel.save == BaseModel.save
  assert (inspect.signature(RNNModel.train_parser) == inspect.signature(BaseModel.train_parser))
