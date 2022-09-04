r"""Test :py:mod:`lmp.util.log` signature."""

import inspect
from inspect import Parameter, Signature

import tensorboardX

import lmp.util.log


def test_module_function():
  """Ensure module function's signature."""
  assert hasattr(lmp.util.log, 'get_tb_logger')
  assert inspect.isfunction(lmp.util.log.get_tb_logger)
  assert inspect.signature(lmp.util.log.get_tb_logger) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='exp_name',
      ),
    ],
    return_annotation=tensorboardX.SummaryWriter,
  )
