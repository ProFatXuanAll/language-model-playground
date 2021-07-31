r"""Test :py:class:`lmp.model._lstm` signature."""

import inspect

from lmp.model._rnn import RNNModel
from lmp.model._lstm import LSTMModel


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(LSTMModel)
    assert not inspect.isabstract(LSTMModel)
    assert issubclass(LSTMModel, RNNModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(LSTMModel.model_name, str)
    assert LSTMModel.model_name == 'LSTM'
    assert LSTMModel.file_name == 'model-{}.pt'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(LSTMModel, '__init__')
    assert inspect.signature(
        LSTMModel.__init__) == inspect.signature(
        RNNModel.__init__
    )


def test_inherent_method():
    r'''Ensure inherent methods' signature are same as base class.'''
    assert (
        inspect.signature(RNNModel.forward)
        ==
        inspect.signature(LSTMModel.forward)
    )

    assert (
        inspect.signature(RNNModel.load)
        ==
        inspect.signature(LSTMModel.load)
    )

    assert (
        inspect.signature(RNNModel.loss_fn)
        ==
        inspect.signature(LSTMModel.loss_fn)
    )

    assert (
        inspect.signature(RNNModel.pred)
        ==
        inspect.signature(LSTMModel.pred)
    )

    assert (
        inspect.signature(RNNModel.ppl)
        ==
        inspect.signature(LSTMModel.ppl)
    )

    assert (
        inspect.signature(RNNModel.save)
        ==
        inspect.signature(LSTMModel.save)
    )

    assert (
        inspect.signature(RNNModel.train_parser)
        ==
        inspect.signature(LSTMModel.train_parser)
    )
