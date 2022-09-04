"""Test :py:mod:`lmp.util.rand` signatures."""

import inspect
from inspect import Parameter, Signature

import lmp.util.rand


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.rand, 'set_seed')
  assert inspect.isfunction(lmp.util.rand.set_seed)
  assert inspect.signature(lmp.util.rand.set_seed) == Signature(
    parameters=[
      Parameter(
        annotation=int,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='seed',
      ),
    ],
    return_annotation=None,
  )
