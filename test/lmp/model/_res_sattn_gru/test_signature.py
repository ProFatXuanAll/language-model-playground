"""Test :py:class:`lmp.model._res_sattn_gru` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

from lmp.model._res_sattn_gru import ResSAttnGRUBlock, ResSAttnGRUModel
from lmp.model._res_sattn_rnn import ResSAttnRNNBlock, ResSAttnRNNModel
from lmp.model._sattn_rnn import SAttnRNNBlock
from lmp.tknzr._base import BaseTknzr


def test_class() -> None:
  """Ensure class signatures."""
  assert inspect.isclass(ResSAttnGRUBlock)
  assert not inspect.isabstract(ResSAttnGRUBlock)
  assert issubclass(ResSAttnGRUBlock, ResSAttnRNNBlock)
  assert issubclass(ResSAttnGRUBlock, SAttnRNNBlock)
  assert inspect.isclass(ResSAttnGRUModel)
  assert not inspect.isabstract(ResSAttnGRUModel)
  assert issubclass(ResSAttnGRUModel, ResSAttnRNNModel)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert isinstance(ResSAttnGRUModel.model_name, str)
  assert ResSAttnGRUModel.model_name == 'res-sattn-GRU'
  assert ResSAttnGRUModel.file_name == 'model-{}.pt'


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(ResSAttnGRUBlock, '__init__')
  assert inspect.signature(ResSAttnGRUBlock.__init__) == Signature(
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

  assert hasattr(ResSAttnGRUModel, '__init__')
  assert inspect.signature(ResSAttnGRUModel.__init__) == Signature(
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
  assert (inspect.signature(ResSAttnRNNModel.forward) == inspect.signature(ResSAttnGRUModel.forward))

  assert (inspect.signature(ResSAttnRNNModel.load) == inspect.signature(ResSAttnGRUModel.load))

  assert (inspect.signature(ResSAttnRNNModel.loss_fn) == inspect.signature(ResSAttnGRUModel.loss_fn))

  assert (inspect.signature(ResSAttnRNNModel.pred) == inspect.signature(ResSAttnGRUModel.pred))

  assert (inspect.signature(ResSAttnRNNModel.ppl) == inspect.signature(ResSAttnGRUModel.ppl))

  assert (inspect.signature(ResSAttnRNNModel.save) == inspect.signature(ResSAttnGRUModel.save))

  assert (inspect.signature(ResSAttnRNNModel.train_parser) == inspect.signature(ResSAttnGRUModel.train_parser))
