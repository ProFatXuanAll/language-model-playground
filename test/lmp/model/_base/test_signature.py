"""Test :py:class:`lmp.model.BaseModel` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import Dict, Optional

import torch

from lmp.model import BaseModel


def test_class() -> None:
  """Ensure abstract class signatures."""
  assert inspect.isclass(BaseModel)
  assert issubclass(BaseModel, torch.nn.Module)
  assert inspect.isabstract(BaseModel)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  assert isinstance(BaseModel.file_name, str)
  assert isinstance(BaseModel.model_name, str)
  assert BaseModel.file_name == 'model-{}.pt'
  assert BaseModel.model_name == 'base'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(BaseModel, 'load')
  assert inspect.ismethod(BaseModel.load)
  assert BaseModel.load.__self__ == BaseModel
  assert inspect.signature(BaseModel.load) == Signature(
    parameters=[
      Parameter(
        name='ckpt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='exp_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
      Parameter(
        name='kwargs',
        kind=Parameter.VAR_KEYWORD,
        annotation=Optional[Dict],
      ),
    ]
  )


def test_instance_method() -> None:
  """Ensure instance methods' signatures."""
  assert hasattr(BaseModel, '__init__')
  assert inspect.isfunction(BaseModel.__init__)
  assert inspect.signature(BaseModel.__init__) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
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
  assert hasattr(BaseModel, 'ppl')
  assert inspect.isfunction(BaseModel.ppl)
  assert inspect.signature(BaseModel.ppl) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='batch_next_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=torch.Tensor,
      ),
      Parameter(
        name='batch_prev_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=torch.Tensor,
      ),
    ],
    return_annotation=float,
  )
  assert hasattr(BaseModel, 'save')
  assert inspect.isfunction(BaseModel.save)
  assert inspect.signature(BaseModel.save) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='ckpt',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='exp_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=None,
  )


def test_abstract_method() -> None:
  """Ensure abstract method's signatures."""
  assert 'forward' in BaseModel.__abstractmethods__
  assert 'loss_fn' in BaseModel.__abstractmethods__
  assert 'pred' in BaseModel.__abstractmethods__


def test_static_method() -> None:
  """Ensure static methods' signatures."""
  assert hasattr(BaseModel, 'train_parser')
  assert inspect.isfunction(BaseModel.train_parser)
  assert inspect.signature(BaseModel.train_parser) == Signature(
    parameters=[
      Parameter(
        name='parser',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=argparse.ArgumentParser,
      ),
    ],
    return_annotation=None,
  )
