"""Test :py:mod:`lmp.model._lstm_2000` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, get_type_hints

import lmp.model._lstm_2000
from lmp.model._lstm_1997 import LSTM1997, LSTM1997Layer
from lmp.tknzr._base import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.model._lstm_2000, 'LSTM2000')
  assert inspect.isclass(lmp.model._lstm_2000.LSTM2000)
  assert issubclass(lmp.model._lstm_2000.LSTM2000, LSTM1997)
  assert not inspect.isabstract(lmp.model._lstm_2000.LSTM2000)

  assert hasattr(lmp.model._lstm_2000, 'LSTM2000Layer')
  assert inspect.isclass(lmp.model._lstm_2000.LSTM2000Layer)
  assert issubclass(lmp.model._lstm_2000.LSTM2000Layer, LSTM1997Layer)
  assert not inspect.isabstract(lmp.model._lstm_2000.LSTM2000Layer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(lmp.model._lstm_2000.LSTM2000)
  assert type_hints['model_name'] == ClassVar[str]
  assert lmp.model._lstm_2000.LSTM2000.model_name == 'LSTM-2000'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.model._lstm_2000.LSTM2000, 'add_CLI_args')
  assert inspect.ismethod(lmp.model._lstm_2000.LSTM2000.add_CLI_args)
  assert lmp.model._lstm_2000.LSTM2000.add_CLI_args.__self__ == lmp.model._lstm_2000.LSTM2000
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.add_CLI_args) == inspect.signature(LSTM1997.add_CLI_args)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.model._lstm_2000.LSTM2000, '__init__')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000.__init__)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.__init__) == Signature(
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
        name='d_blk',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='d_emb',
      ),
      Parameter(
        annotation=float,
        default=1.0,
        kind=Parameter.KEYWORD_ONLY,
        name='init_fb',
      ),
      Parameter(
        annotation=float,
        default=-1.0,
        kind=Parameter.KEYWORD_ONLY,
        name='init_ib',
      ),
      Parameter(
        annotation=float,
        default=-0.1,
        kind=Parameter.KEYWORD_ONLY,
        name='init_lower',
      ),
      Parameter(
        annotation=float,
        default=-1.0,
        kind=Parameter.KEYWORD_ONLY,
        name='init_ob',
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
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='n_blk',
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
        name='p_emb',
      ),
      Parameter(
        annotation=float,
        default=0.0,
        kind=Parameter.KEYWORD_ONLY,
        name='p_hid',
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

  assert hasattr(lmp.model._lstm_2000.LSTM2000, 'cal_loss')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000.cal_loss)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.cal_loss) == inspect.signature(LSTM1997.cal_loss)

  assert hasattr(lmp.model._lstm_2000.LSTM2000, 'forward')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000.forward)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.forward) == inspect.signature(LSTM1997.forward)

  assert hasattr(lmp.model._lstm_2000.LSTM2000, 'params_init')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000.params_init)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.params_init) == inspect.signature(LSTM1997.params_init)

  assert hasattr(lmp.model._lstm_2000.LSTM2000, 'pred')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000.pred)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.pred) == inspect.signature(LSTM1997.pred)

  assert hasattr(lmp.model._lstm_2000.LSTM2000Layer, '__init__')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000Layer.__init__)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000Layer.__init__) == Signature(
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
        name='d_blk',
      ),
      Parameter(
        annotation=int,
        default=1,
        kind=Parameter.KEYWORD_ONLY,
        name='in_feat',
      ),
      Parameter(
        annotation=float,
        default=1.0,
        kind=Parameter.KEYWORD_ONLY,
        name='init_fb',
      ),
      Parameter(
        annotation=float,
        default=-1.0,
        kind=Parameter.KEYWORD_ONLY,
        name='init_ib',
      ),
      Parameter(
        annotation=float,
        default=-0.1,
        kind=Parameter.KEYWORD_ONLY,
        name='init_lower',
      ),
      Parameter(
        annotation=float,
        default=-1.0,
        kind=Parameter.KEYWORD_ONLY,
        name='init_ob',
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
        name='n_blk',
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

  assert hasattr(lmp.model._lstm_2000.LSTM2000Layer, 'forward')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000Layer.forward)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000Layer.forward) == inspect.signature(LSTM1997Layer.forward)

  assert hasattr(lmp.model._lstm_2000.LSTM2000Layer, 'params_init')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000Layer.params_init)
  assert (
    inspect.signature(lmp.model._lstm_2000.LSTM2000Layer.params_init) == inspect.signature(LSTM1997Layer.params_init)
  )
