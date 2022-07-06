"""Test :py:mod:`lmp.model._lstm_1997` signatures."""

import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, get_type_hints

import lmp.model._lstm_1997
from lmp.model._base import BaseModel
from lmp.tknzr._base import BaseTknzr


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.model._lstm_1997, 'LSTM1997')
  assert inspect.isclass(lmp.model._lstm_1997.LSTM1997)
  assert issubclass(lmp.model._lstm_1997.LSTM1997, BaseModel)
  assert not inspect.isabstract(lmp.model._lstm_1997.LSTM1997)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(lmp.model._lstm_1997.LSTM1997)
  assert type_hints['model_name'] == ClassVar[str]
  assert lmp.model._lstm_1997.LSTM1997.model_name == 'LSTM-1997'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.model._lstm_1997.LSTM1997, 'add_CLI_args')
  assert inspect.ismethod(lmp.model._lstm_1997.LSTM1997.add_CLI_args)
  assert lmp.model._lstm_1997.LSTM1997.add_CLI_args.__self__ == lmp.model._lstm_1997.LSTM1997
  assert inspect.signature(lmp.model._lstm_1997.LSTM1997.add_CLI_args) == inspect.signature(BaseModel.add_CLI_args)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.model._lstm_1997.LSTM1997, '__init__')
  assert inspect.isfunction(lmp.model._lstm_1997.LSTM1997.__init__)
  assert inspect.signature(lmp.model._lstm_1997.LSTM1997.__init__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='d_blk',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='d_emb',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='n_blk',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='p_emb',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=float,
      ),
      Parameter(
        name='p_hid',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=float,
      ),
      Parameter(
        name='tknzr',
        kind=Parameter.KEYWORD_ONLY,
        default=Parameter.empty,
        annotation=BaseTknzr,
      ),
      Parameter(
        name='kwargs',
        kind=Parameter.VAR_KEYWORD,
        annotation=Any,
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.model._lstm_1997.LSTM1997, 'forward')
  assert inspect.isfunction(lmp.model._lstm_1997.LSTM1997.forward)
  assert inspect.signature(lmp.model._lstm_1997.LSTM1997.forward) == inspect.signature(BaseModel.forward)

  assert hasattr(lmp.model._lstm_1997.LSTM1997, 'loss')
  assert inspect.isfunction(lmp.model._lstm_1997.LSTM1997.loss)
  assert inspect.signature(lmp.model._lstm_1997.LSTM1997.loss) == inspect.signature(BaseModel.loss)

  assert hasattr(lmp.model._lstm_1997.LSTM1997, 'params_init')
  assert inspect.isfunction(lmp.model._lstm_1997.LSTM1997.params_init)
  assert inspect.signature(lmp.model._lstm_1997.LSTM1997.params_init) == inspect.signature(BaseModel.params_init)

  assert hasattr(lmp.model._lstm_1997.LSTM1997, 'pred')
  assert inspect.isfunction(lmp.model._lstm_1997.LSTM1997.pred)
  assert inspect.signature(lmp.model._lstm_1997.LSTM1997.pred) == inspect.signature(BaseModel.pred)
