"""Test :py:mod:`lmp.model._trans_enc` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, Optional, Tuple, get_type_hints

import torch

import lmp.model._trans_enc
from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.model._trans_enc, 'MultiHeadAttnLayer')
  assert inspect.isclass(lmp.model._trans_enc.MultiHeadAttnLayer)
  assert issubclass(lmp.model._trans_enc.MultiHeadAttnLayer, torch.nn.Module)
  assert not inspect.isabstract(lmp.model._trans_enc.MultiHeadAttnLayer)

  assert hasattr(lmp.model._trans_enc, 'PosEncLayer')
  assert inspect.isclass(lmp.model._trans_enc.PosEncLayer)
  assert issubclass(lmp.model._trans_enc.PosEncLayer, torch.nn.Module)
  assert not inspect.isabstract(lmp.model._trans_enc.PosEncLayer)

  assert hasattr(lmp.model._trans_enc, 'TransEnc')
  assert inspect.isclass(lmp.model._trans_enc.TransEnc)
  assert issubclass(lmp.model._trans_enc.TransEnc, BaseModel)
  assert not inspect.isabstract(lmp.model._trans_enc.TransEnc)

  assert hasattr(lmp.model._trans_enc, 'TransEncLayer')
  assert inspect.isclass(lmp.model._trans_enc.TransEncLayer)
  assert issubclass(lmp.model._trans_enc.TransEncLayer, torch.nn.Module)
  assert not inspect.isabstract(lmp.model._trans_enc.TransEncLayer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(lmp.model._trans_enc.TransEnc)
  assert type_hints['model_name'] == ClassVar[str]
  assert lmp.model._trans_enc.TransEnc.model_name == 'Transformer-encoder'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.model._trans_enc.TransEnc, 'add_CLI_args')
  assert inspect.ismethod(lmp.model._trans_enc.TransEnc.add_CLI_args)
  assert lmp.model._trans_enc.TransEnc.add_CLI_args.__self__ == lmp.model._trans_enc.TransEnc
  assert inspect.signature(lmp.model._trans_enc.TransEnc.add_CLI_args) == inspect.signature(BaseModel.add_CLI_args)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.model._trans_enc.MultiHeadAttnLayer, '__init__')
  assert inspect.isfunction(lmp.model._trans_enc.MultiHeadAttnLayer.__init__)
  assert inspect.signature(lmp.model._trans_enc.MultiHeadAttnLayer.__init__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_k',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_model',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_v',
      ),
      Parameter(
        annotation=float,
        default=-0.1,
        kind=Parameter.KEYWORD_ONLY,
        name='init_lower',
      ),
      Parameter(
        annotation=float,
        default=0.1,
        kind=Parameter.KEYWORD_ONLY,
        name='init_upper',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='n_head',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.model._trans_enc.MultiHeadAttnLayer, 'forward')
  assert inspect.isfunction(lmp.model._trans_enc.MultiHeadAttnLayer.forward)
  assert inspect.signature(lmp.model._trans_enc.MultiHeadAttnLayer.forward) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='k',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='mask',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='q',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='v',
      ),
    ],
    return_annotation=torch.Tensor,
  )

  assert hasattr(lmp.model._trans_enc.MultiHeadAttnLayer, 'params_init')
  assert inspect.isfunction(lmp.model._trans_enc.MultiHeadAttnLayer.params_init)
  assert (
    inspect.signature(lmp.model._trans_enc.MultiHeadAttnLayer.params_init) == inspect.signature(BaseModel.params_init)
  )

  assert hasattr(lmp.model._trans_enc.PosEncLayer, '__init__')
  assert inspect.isfunction(lmp.model._trans_enc.PosEncLayer.__init__)
  assert inspect.signature(lmp.model._trans_enc.PosEncLayer.__init__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_emb',
      ),
      Parameter(
        annotation=int,
        default=512,
        kind=Parameter.KEYWORD_ONLY,
        name='max_seq_len',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.model._trans_enc.PosEncLayer, 'forward')
  assert inspect.isfunction(lmp.model._trans_enc.PosEncLayer.forward)
  assert inspect.signature(lmp.model._trans_enc.PosEncLayer.forward) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='seq_len',
      ),
    ],
    return_annotation=torch.Tensor,
  )

  assert hasattr(lmp.model._trans_enc.PosEncLayer, 'params_init')
  assert inspect.isfunction(lmp.model._trans_enc.PosEncLayer.params_init)
  assert inspect.signature(lmp.model._trans_enc.PosEncLayer.params_init) == inspect.signature(BaseModel.params_init)

  assert hasattr(lmp.model._trans_enc.TransEnc, '__init__')
  assert inspect.isfunction(lmp.model._trans_enc.TransEnc.__init__)
  assert inspect.signature(lmp.model._trans_enc.TransEnc.__init__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_ff',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_k',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_model',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_v',
      ),
      Parameter(
        annotation=float,
        default=-0.1,
        kind=Parameter.KEYWORD_ONLY,
        name='init_lower',
      ),
      Parameter(
        annotation=float,
        default=0.1,
        kind=Parameter.KEYWORD_ONLY,
        name='init_upper',
      ),
      Parameter(
        annotation=float,
        default=0.0,
        kind=Parameter.KEYWORD_ONLY,
        name='label_smoothing',
      ),
      Parameter(
        annotation=int,
        default=512,
        kind=Parameter.KEYWORD_ONLY,
        name='max_seq_len',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='n_head',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='n_lyr',
      ),
      Parameter(
        annotation=float,
        default=0.0,
        kind=Parameter.KEYWORD_ONLY,
        name='p',
      ),
      Parameter(
        annotation=BaseTknzr,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='tknzr',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.model._trans_enc.TransEnc, 'cal_loss')
  assert inspect.isfunction(lmp.model._trans_enc.TransEnc.cal_loss)
  assert inspect.signature(lmp.model._trans_enc.TransEnc.cal_loss) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_cur_tkids',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_next_tkids',
      ),
      Parameter(
        annotation=Optional[torch.Tensor],
        default=None,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_prev_states',
      ),
    ],
    return_annotation=Tuple[torch.Tensor, torch.Tensor],
  )

  assert hasattr(lmp.model._trans_enc.TransEnc, 'create_mask')
  assert inspect.isfunction(lmp.model._trans_enc.TransEnc.create_mask)
  assert inspect.signature(lmp.model._trans_enc.TransEnc.create_mask) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_tkids',
      ),
    ],
    return_annotation=torch.Tensor,
  )

  assert hasattr(lmp.model._trans_enc.TransEnc, 'forward')
  assert inspect.isfunction(lmp.model._trans_enc.TransEnc.forward)
  assert inspect.signature(lmp.model._trans_enc.TransEnc.forward) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_cur_tkids',
      ),
      Parameter(
        annotation=Optional[torch.Tensor],
        default=None,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_prev_states',
      ),
    ],
    return_annotation=Tuple[torch.Tensor, torch.Tensor],
  )

  assert hasattr(lmp.model._trans_enc.TransEnc, 'params_init')
  assert inspect.isfunction(lmp.model._trans_enc.TransEnc.params_init)
  assert inspect.signature(lmp.model._trans_enc.TransEnc.params_init) == inspect.signature(BaseModel.params_init)

  assert hasattr(lmp.model._trans_enc.TransEnc, 'pred')
  assert inspect.isfunction(lmp.model._trans_enc.TransEnc.pred)
  assert inspect.signature(lmp.model._trans_enc.TransEnc.pred) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_cur_tkids',
      ),
      Parameter(
        annotation=Optional[torch.Tensor],
        default=None,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_prev_states',
      ),
    ],
    return_annotation=Tuple[torch.Tensor, torch.Tensor],
  )

  assert hasattr(lmp.model._trans_enc.TransEncLayer, '__init__')
  assert inspect.isfunction(lmp.model._trans_enc.TransEncLayer.__init__)
  assert inspect.signature(lmp.model._trans_enc.TransEncLayer.__init__) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_ff',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_k',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_model',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_v',
      ),
      Parameter(
        annotation=float,
        default=-0.1,
        kind=Parameter.KEYWORD_ONLY,
        name='init_lower',
      ),
      Parameter(
        annotation=float,
        default=0.1,
        kind=Parameter.KEYWORD_ONLY,
        name='init_upper',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='n_head',
      ),
      Parameter(
        annotation=float,
        default=0.0,
        kind=Parameter.KEYWORD_ONLY,
        name='p',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.model._trans_enc.TransEncLayer, 'forward')
  assert inspect.isfunction(lmp.model._trans_enc.TransEncLayer.forward)
  assert inspect.signature(lmp.model._trans_enc.TransEncLayer.forward) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='mask',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='x',
      ),
    ],
    return_annotation=torch.Tensor,
  )

  assert hasattr(lmp.model._trans_enc.TransEncLayer, 'params_init')
  assert inspect.isfunction(lmp.model._trans_enc.TransEncLayer.params_init)
  assert inspect.signature(lmp.model._trans_enc.TransEncLayer.params_init) == inspect.signature(BaseModel.params_init)
