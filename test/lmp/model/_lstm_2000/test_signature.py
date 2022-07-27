"""Test :py:mod:`lmp.model._lstm_2000` signatures."""

import inspect
from typing import ClassVar, get_type_hints

import lmp.model._lstm_2000
from lmp.model._lstm_1997 import LSTM1997, LSTM1997Layer


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.model._lstm_2000, 'LSTM2000Layer')
  assert inspect.isclass(lmp.model._lstm_2000.LSTM2000Layer)
  assert issubclass(lmp.model._lstm_2000.LSTM2000Layer, LSTM1997Layer)
  assert not inspect.isabstract(lmp.model._lstm_2000.LSTM2000Layer)
  assert hasattr(lmp.model._lstm_2000, 'LSTM2000')
  assert inspect.isclass(lmp.model._lstm_2000.LSTM2000)
  assert issubclass(lmp.model._lstm_2000.LSTM2000, LSTM1997)
  assert not inspect.isabstract(lmp.model._lstm_2000.LSTM2000)


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
  assert hasattr(lmp.model._lstm_2000.LSTM2000Layer, '__init__')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000Layer.__init__)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000Layer.__init__) == inspect.signature(LSTM1997Layer.__init__)

  assert hasattr(lmp.model._lstm_2000.LSTM2000Layer, 'forward')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000Layer.forward)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000Layer.forward) == inspect.signature(LSTM1997Layer.forward)

  assert hasattr(lmp.model._lstm_2000.LSTM2000Layer, 'params_init')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000Layer.params_init)
  assert (
    inspect.signature(lmp.model._lstm_2000.LSTM2000Layer.params_init) == inspect.signature(LSTM1997Layer.params_init)
  )

  assert hasattr(lmp.model._lstm_2000.LSTM2000, '__init__')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000.__init__)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.__init__) == inspect.signature(LSTM1997.__init__)

  assert hasattr(lmp.model._lstm_2000.LSTM2000, 'forward')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000.forward)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.forward) == inspect.signature(LSTM1997.forward)

  assert hasattr(lmp.model._lstm_2000.LSTM2000, 'params_init')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000.params_init)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.params_init) == inspect.signature(LSTM1997.params_init)

  assert hasattr(lmp.model._lstm_2000.LSTM2000, 'pred')
  assert inspect.isfunction(lmp.model._lstm_2000.LSTM2000.pred)
  assert inspect.signature(lmp.model._lstm_2000.LSTM2000.pred) == inspect.signature(LSTM1997.pred)
