r"""Test :py:class:`lmp.model` signature."""

import inspect

from lmp.model import GRUModel, RNNModel


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(GRUModel)
    assert not inspect.isabstract(GRUModel)
    assert issubclass(GRUModel, RNNModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(GRUModel.model_name, str)
    assert GRUModel.model_name == 'GRU'
    assert GRUModel.file_name == RNNModel.file_name


def test_inherent_method():
    r'''Ensure inherent methods' signature are same as base class.'''
    assert (
        inspect.signature(GRUModel.__init__)
        ==
        inspect.signature(RNNModel.__init__)
    )
    assert GRUModel.forward == RNNModel.forward
    assert (
        inspect.signature(GRUModel.load)
        ==
        inspect.signature(RNNModel.load)
    )
    assert GRUModel.loss_fn == RNNModel.loss_fn
    assert GRUModel.pred == RNNModel.pred
    assert GRUModel.ppl == RNNModel.ppl
    assert GRUModel.save == RNNModel.save
    assert GRUModel.train_parser == RNNModel.train_parser
