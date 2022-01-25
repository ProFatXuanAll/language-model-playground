"""Test :py:class:`lmp.model.BaseModel` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, List, Optional, Tuple, get_type_hints

import torch

from lmp.model import BaseModel


def test_class() -> None:
  """Ensure abstract class signatures."""
  assert inspect.isclass(BaseModel)
  assert issubclass(BaseModel, torch.nn.Module)
  assert inspect.isabstract(BaseModel)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(BaseModel)
  assert type_hints['model_name'] == ClassVar[str]
  assert BaseModel.model_name == 'base'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(BaseModel, 'train_parser')
  assert inspect.ismethod(BaseModel.train_parser)
  assert BaseModel.train_parser.__self__ == BaseModel
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
        annotation=Any,
      ),
    ],
    return_annotation=Signature.empty,
  )
  assert hasattr(BaseModel, 'forward')
  assert inspect.isfunction(BaseModel.forward)
  assert 'forward' in BaseModel.__abstractmethods__
  assert inspect.signature(BaseModel.forward) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='batch_cur_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=torch.Tensor,
      ),
      Parameter(
        name='batch_next_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=torch.Tensor,
      ),
    ],
    return_annotation=torch.Tensor,
  )
  assert hasattr(BaseModel, 'pred')
  assert inspect.isfunction(BaseModel.pred)
  assert 'pred' in BaseModel.__abstractmethods__
  assert inspect.signature(BaseModel.pred) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
      Parameter(
        name='batch_cur_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=torch.Tensor,
      ),
      Parameter(
        name='batch_prev_states',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=Optional[List[torch.Tensor]],
        default=None,
      ),
    ],
    return_annotation=Tuple[torch.Tensor, List[torch.Tensor]],
  )
