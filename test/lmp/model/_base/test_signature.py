"""Test :py:mod:`lmp.model._base` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, List, Optional, Tuple, get_type_hints

import torch

import lmp.model._base


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.model._base, 'BaseModel')
  assert inspect.isclass(lmp.model._base.BaseModel)
  assert issubclass(lmp.model._base.BaseModel, torch.nn.Module)
  assert inspect.isabstract(lmp.model._base.BaseModel)


def test_class_attribute() -> None:
  """Ensure class attributes' signatures."""
  type_hints = get_type_hints(lmp.model._base.BaseModel)
  assert type_hints['model_name'] == ClassVar[str]
  assert lmp.model._base.BaseModel.model_name == 'base'


def test_class_method() -> None:
  """Ensure class methods' signatures."""
  assert hasattr(lmp.model._base.BaseModel, 'add_CLI_args')
  assert inspect.ismethod(lmp.model._base.BaseModel.add_CLI_args)
  assert 'add_CLI_args' in lmp.model._base.BaseModel.__abstractmethods__
  assert lmp.model._base.BaseModel.add_CLI_args.__self__ == lmp.model._base.BaseModel
  assert inspect.signature(lmp.model._base.BaseModel.add_CLI_args) == Signature(
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
  assert hasattr(lmp.model._base.BaseModel, '__init__')
  assert inspect.isfunction(lmp.model._base.BaseModel.__init__)
  assert inspect.signature(lmp.model._base.BaseModel.__init__) == Signature(
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
  assert hasattr(lmp.model._base.BaseModel, 'forward')
  assert inspect.isfunction(lmp.model._base.BaseModel.forward)
  assert 'forward' in lmp.model._base.BaseModel.__abstractmethods__
  assert inspect.signature(lmp.model._base.BaseModel.forward) == Signature(
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
  assert hasattr(lmp.model._base.BaseModel, 'params_init')
  assert inspect.isfunction(lmp.model._base.BaseModel.params_init)
  assert 'params_init' in lmp.model._base.BaseModel.__abstractmethods__
  assert inspect.signature(lmp.model._base.BaseModel.params_init) == Signature(
    parameters=[
      Parameter(
        name='self',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
      ),
    ],
    return_annotation=None,
  )
  assert hasattr(lmp.model._base.BaseModel, 'pred')
  assert inspect.isfunction(lmp.model._base.BaseModel.pred)
  assert 'pred' in lmp.model._base.BaseModel.__abstractmethods__
  assert inspect.signature(lmp.model._base.BaseModel.pred) == Signature(
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
