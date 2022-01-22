"""Test :py:mod:`lmp.util.metric` signatures."""

import inspect
from inspect import Parameter, Signature

import torch

import lmp.util.metric


def test_module_function() -> None:
  """Ensure module function's signatures."""
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
    ],
    return_annotation=torch.Tensor,
  )
