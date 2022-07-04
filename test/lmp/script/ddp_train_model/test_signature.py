"""Test :py:mod:`lmp.script.ddp_train_model` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import List

import lmp.script.ddp_train_model


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.script.ddp_train_model, 'HOST_RANK')
  assert lmp.script.ddp_train_model.HOST_RANK == 0


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.script.ddp_train_model, 'parse_args')
  assert inspect.isfunction(lmp.script.ddp_train_model.parse_args)
  assert inspect.signature(lmp.script.ddp_train_model.parse_args) == Signature(
    parameters=[
      Parameter(
        name='argv',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[str],
      ),
    ],
    return_annotation=argparse.Namespace,
  )
  assert hasattr(lmp.script.ddp_train_model, 'main')
  assert inspect.isfunction(lmp.script.ddp_train_model.main)
  assert inspect.signature(lmp.script.ddp_train_model.main) == Signature(
    parameters=[
      Parameter(
        name='argv',
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=List[str],
      ),
    ],
    return_annotation=None,
  )