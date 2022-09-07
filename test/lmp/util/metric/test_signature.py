"""Test :py:mod:`lmp.util.metric` signatures."""

import inspect
from inspect import Parameter, Signature

import torch

import lmp.util.metric


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.metric, 'nll')
  assert inspect.isfunction(lmp.util.metric.nll)
  assert inspect.signature(lmp.util.metric.nll) == Signature(
    parameters=[
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_tkids',
      ),
      Parameter(
        annotation=torch.Tensor,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='batch_tkids_pd',
      ),
      Parameter(
        annotation=bool,
        default=True,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='use_log2',
      ),
    ],
    return_annotation=torch.Tensor,
  )
