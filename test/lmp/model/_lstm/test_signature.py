r"""Test :py:class:`lmp.model` signature."""

import inspect

from lmp.model import LSTMModel, RNNModel


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(LSTMModel)
    assert not inspect.isabstract(LSTMModel)
    assert issubclass(LSTMModel, RNNModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(LSTMModel.model_name, str)
    assert LSTMModel.model_name == 'LSTM'
    assert LSTMModel.file_name == RNNModel.file_name


def test_inherent_method():
    r'''Ensure inherent methods' signature are same as base class.'''
    assert (
        inspect.signature(LSTMModel.__init__)
        ==
        inspect.signature(RNNModel.__init__)
    )
    assert LSTMModel.forward == RNNModel.forward
    assert (
        inspect.signature(LSTMModel.load)
        ==
        inspect.signature(RNNModel.load)
    )
    assert LSTMModel.loss_fn == RNNModel.loss_fn
    assert LSTMModel.pred == RNNModel.pred
    assert LSTMModel.ppl == RNNModel.ppl
    assert LSTMModel.save == RNNModel.save
    assert LSTMModel.train_parser == RNNModel.train_parser
