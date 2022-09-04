"""Test :py:mod:`lmp.util.optim` signatures."""

import inspect
from inspect import Parameter, Signature

import torch

import lmp.util.optim
from lmp.model import BaseModel


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.optim, 'get_optimizer')
  assert inspect.isfunction(lmp.util.optim.get_optimizer)
  assert inspect.signature(lmp.util.optim.get_optimizer) == Signature(
    parameters=[
      Parameter(
        annotation=float,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='beta1',
      ),
      Parameter(
        annotation=float,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='beta2',
      ),
      Parameter(
        annotation=float,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='eps',
      ),
      Parameter(
        annotation=float,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='lr',
      ),
      Parameter(
        annotation=BaseModel,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='model',
      ),
      Parameter(
        annotation=float,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='weight_decay',
      ),
    ],
    return_annotation=torch.optim.AdamW,
  )

  assert hasattr(lmp.util.optim, 'get_scheduler')
  assert inspect.isfunction(lmp.util.optim.get_scheduler)
  assert inspect.signature(lmp.util.optim.get_scheduler) == Signature(
    parameters=[
      Parameter(
        annotation=torch.optim.AdamW,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='optim',
      ),
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='total_step',
      ),
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='warmup_step',
      ),
    ],
    return_annotation=torch.optim.lr_scheduler.LambdaLR,
  )
