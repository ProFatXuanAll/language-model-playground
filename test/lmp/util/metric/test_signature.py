"""Test :py:mod:`lmp.util.metric` signatures."""

import inspect
from inspect import Parameter, Signature

import torch

import lmp.util.metric


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.metric, 'ppl')
  assert inspect.isfunction(lmp.util.metric.ppl)
  assert inspect.signature(lmp.util.metric.ppl) == Signature(
    parameters=[
      Parameter(
        name='batch_tkids',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=torch.Tensor,
      ),
      Parameter(
        name='batch_tkids_pd',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=torch.Tensor,
      ),
      Parameter(
        name='eos_tkid',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=int,
      ),
      Parameter(
        name='pad_tkid',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        annotation=int,
      ),
    ],
    return_annotation=torch.Tensor,
  )
