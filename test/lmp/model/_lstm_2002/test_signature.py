"""Test :py:mod:`lmp.model._lstm_2002` signatures."""

import inspect
from typing import ClassVar, get_type_hints

import lmp.model._lstm_2002
from lmp.model._lstm_2000 import LSTM2000, LSTM2000Layer


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.model._lstm_2002, 'LSTM2002')
  assert inspect.isclass(lmp.model._lstm_2002.LSTM2002)
  assert issubclass(lmp.model._lstm_2002.LSTM2002, LSTM2000)
  assert not inspect.isabstract(lmp.model._lstm_2002.LSTM2002)

  assert hasattr(lmp.model._lstm_2002, 'LSTM2002Layer')
  assert inspect.isclass(lmp.model._lstm_2002.LSTM2002Layer)
  assert issubclass(lmp.model._lstm_2002.LSTM2002Layer, LSTM2000Layer)
  assert not inspect.isabstract(lmp.model._lstm_2002.LSTM2002Layer)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(lmp.model._lstm_2002.LSTM2002)
  assert type_hints['model_name'] == ClassVar[str]
  assert lmp.model._lstm_2002.LSTM2002.model_name == 'LSTM-2002'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.model._lstm_2002.LSTM2002, 'add_CLI_args')
  assert inspect.ismethod(lmp.model._lstm_2002.LSTM2002.add_CLI_args)
  assert lmp.model._lstm_2002.LSTM2002.add_CLI_args.__self__ == lmp.model._lstm_2002.LSTM2002
  assert inspect.signature(lmp.model._lstm_2002.LSTM2002.add_CLI_args) == inspect.signature(LSTM2000.add_CLI_args)


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(lmp.model._lstm_2002.LSTM2002, '__init__')
  assert inspect.isfunction(lmp.model._lstm_2002.LSTM2002.__init__)
  assert inspect.signature(lmp.model._lstm_2002.LSTM2002.__init__) == inspect.signature(LSTM2000.__init__)

  assert hasattr(lmp.model._lstm_2002.LSTM2002, 'cal_loss')
  assert inspect.isfunction(lmp.model._lstm_2002.LSTM2002.cal_loss)
  assert inspect.signature(lmp.model._lstm_2002.LSTM2002.cal_loss) == inspect.signature(LSTM2000.cal_loss)

  assert hasattr(lmp.model._lstm_2002.LSTM2002, 'forward')
  assert inspect.isfunction(lmp.model._lstm_2002.LSTM2002.forward)
  assert inspect.signature(lmp.model._lstm_2002.LSTM2002.forward) == inspect.signature(LSTM2000.forward)

  assert hasattr(lmp.model._lstm_2002.LSTM2002, 'params_init')
  assert inspect.isfunction(lmp.model._lstm_2002.LSTM2002.params_init)
  assert inspect.signature(lmp.model._lstm_2002.LSTM2002.params_init) == inspect.signature(LSTM2000.params_init)

  assert hasattr(lmp.model._lstm_2002.LSTM2002, 'pred')
  assert inspect.isfunction(lmp.model._lstm_2002.LSTM2002.pred)
  assert inspect.signature(lmp.model._lstm_2002.LSTM2002.pred) == inspect.signature(LSTM2000.pred)

  assert hasattr(lmp.model._lstm_2002.LSTM2002Layer, '__init__')
  assert inspect.isfunction(lmp.model._lstm_2002.LSTM2002Layer.__init__)
  assert inspect.signature(lmp.model._lstm_2002.LSTM2002Layer.__init__) == inspect.signature(LSTM2000Layer.__init__)

  assert hasattr(lmp.model._lstm_2002.LSTM2002Layer, 'forward')
  assert inspect.isfunction(lmp.model._lstm_2002.LSTM2002Layer.forward)
  assert inspect.signature(lmp.model._lstm_2002.LSTM2002Layer.forward) == inspect.signature(LSTM2000Layer.forward)

  assert hasattr(lmp.model._lstm_2002.LSTM2002Layer, 'params_init')
  assert inspect.isfunction(lmp.model._lstm_2002.LSTM2002Layer.params_init)
  assert (
    inspect.signature(lmp.model._lstm_2002.LSTM2002Layer.params_init) == inspect.signature(LSTM2000Layer.params_init)
  )
