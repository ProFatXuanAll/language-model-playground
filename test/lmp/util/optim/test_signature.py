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
        name='beta1',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=float,
      ),
      Parameter(
        name='beta2',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=float,
      ),
      Parameter(
        name='eps',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=float,
      ),
      Parameter(
        name='lr',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=float,
      ),
      Parameter(
        name='model',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=BaseModel,
      ),
      Parameter(
        name='wd',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=float,
      ),
    ],
    return_annotation=torch.optim.AdamW,
  )
  assert hasattr(lmp.util.optim, 'get_scheduler')
  assert inspect.isfunction(lmp.util.optim.get_scheduler)
  assert inspect.signature(lmp.util.optim.get_scheduler) == Signature(
    parameters=[
      Parameter(
        name='optim',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=torch.optim.AdamW,
      ),
      Parameter(
        name='total_step',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
      ),
      Parameter(
        name='warmup_step',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
      ),
    ],
    return_annotation=torch.optim.lr_scheduler.LambdaLR,
  )
