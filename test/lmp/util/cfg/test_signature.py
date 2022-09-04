"""Test :py:mod:`lmp.util.cfg` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature

import lmp.util.cfg


def test_module_attribute():
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.util.cfg, 'FILE_NAME')
  assert lmp.util.cfg.FILE_NAME == 'cfg.json'


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.util.cfg, 'load')
  assert inspect.isfunction(lmp.util.cfg.load)
  assert inspect.signature(lmp.util.cfg.load) == Signature(
    parameters=[
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='exp_name',
      ),
    ],
    return_annotation=argparse.Namespace,
  )

  assert hasattr(lmp.util.cfg, 'save')
  assert inspect.isfunction(lmp.util.cfg.save)
  assert inspect.signature(lmp.util.cfg.save) == Signature(
    parameters=[
      Parameter(
        annotation=argparse.Namespace,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='args',
      ),
      Parameter(
        annotation=str,
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='exp_name',
      ),
    ],
    return_annotation=None,
  )
