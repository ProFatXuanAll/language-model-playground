r"""Test :py:class:`lmp.model._gru` signature."""

import inspect

from lmp.model._rnn import RNNModel
from lmp.model._gru import GRUModel


def test_class():
    r"""Ensure class signature."""
    assert inspect.isclass(GRUModel)
    assert not inspect.isabstract(GRUModel)
    assert issubclass(GRUModel, RNNModel)


def test_class_attribute():
    r"""Ensure class attributes' signature."""
    assert isinstance(GRUModel.model_name, str)
    assert GRUModel.model_name == 'GRU'
    assert GRUModel.file_name == 'model-{}.pt'


def test_instance_method():
    r"""Ensure instance methods' signature."""
    assert hasattr(GRUModel, '__init__')
    assert inspect.signature(
        GRUModel.__init__) == inspect.signature(
        RNNModel.__init__
    )


def test_inherent_method():
    r'''Ensure inherent methods' signature are same as base class.'''
    assert (
        inspect.signature(RNNModel.forward)
        ==
        inspect.signature(GRUModel.forward)
    )

    assert (
        inspect.signature(RNNModel.load)
        ==
        inspect.signature(GRUModel.load)
    )

    assert (
        inspect.signature(RNNModel.loss_fn)
        ==
        inspect.signature(GRUModel.loss_fn)
    )

    assert (
        inspect.signature(RNNModel.pred)
        ==
        inspect.signature(GRUModel.pred)
    )

    assert (
        inspect.signature(RNNModel.ppl)
        ==
        inspect.signature(GRUModel.ppl)
    )

    assert (
        inspect.signature(RNNModel.save)
        ==
        inspect.signature(GRUModel.save)
    )

    assert (
        inspect.signature(RNNModel.train_parser)
        ==
        inspect.signature(GRUModel.train_parser)
    )
