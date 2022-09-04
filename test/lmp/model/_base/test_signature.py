"""Test :py:mod:`lmp.model._base` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import Any, ClassVar, Tuple, get_type_hints

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
        annotation=argparse.ArgumentParser,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='parser',
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
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=Any,
        default=Parameter.empty,
        kind=Parameter.VAR_KEYWORD,
        name='kwargs',
      ),
    ],
    return_annotation=Signature.empty,
  )

  assert hasattr(lmp.model._base.BaseModel, 'cal_loss')
  assert inspect.isfunction(lmp.model._base.BaseModel.cal_loss)
  assert 'cal_loss' in lmp.model._base.BaseModel.__abstractmethods__
  assert inspect.signature(lmp.model._base.BaseModel.cal_loss) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_cur_tkids',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_next_tkids',
      ),
      Parameter(
        annotation=Any,
        default=None,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_prev_states',
      ),
    ],
    return_annotation=Tuple[torch.Tensor, Any],
  )

  assert hasattr(lmp.model._base.BaseModel, 'forward')
  assert inspect.isfunction(lmp.model._base.BaseModel.forward)
  assert 'forward' in lmp.model._base.BaseModel.__abstractmethods__
  assert inspect.signature(lmp.model._base.BaseModel.forward) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_cur_tkids',
      ),
      Parameter(
        annotation=Any,
        default=None,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_prev_states',
      ),
    ],
    return_annotation=Tuple[torch.Tensor, Any],
  )

  assert hasattr(lmp.model._base.BaseModel, 'params_init')
  assert inspect.isfunction(lmp.model._base.BaseModel.params_init)
  assert 'params_init' in lmp.model._base.BaseModel.__abstractmethods__
  assert inspect.signature(lmp.model._base.BaseModel.params_init) == Signature(
    parameters=[
      Parameter(
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
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
        annotation=Parameter.empty,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='self',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_cur_tkids',
      ),
      Parameter(
        annotation=Any,
        default=None,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_prev_states',
      ),
    ],
    return_annotation=Tuple[torch.Tensor, Any],
  )
