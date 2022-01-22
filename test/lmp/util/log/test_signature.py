r"""Test :py:mod:`lmp.util.log` signature."""

import inspect
from inspect import Parameter, Signature

import tensorboardX

import lmp.util.log


def test_module_function():
  """Ensure module function's signature."""
  assert inspect.isfunction(lmp.util.log.get_tb_logger)
  assert inspect.signature(lmp.util.log.get_tb_logger) == Signature(
    parameters=[
      Parameter(
        name='exp_name',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=str,
      ),
    ],
    return_annotation=tensorboardX.SummaryWriter,
  )
