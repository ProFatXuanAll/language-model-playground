"""Test :py:mod:`lmp.script.eval_txt_ppl` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import List

import lmp.script.eval_txt_ppl


def test_module_method() -> None:
  """Ensure module methods' signatures."""
  assert hasattr(lmp.script.eval_txt_ppl, 'parse_args')
  assert inspect.isfunction(lmp.script.eval_txt_ppl.parse_args)
  assert inspect.signature(lmp.script.eval_txt_ppl.parse_args) == Signature(
    parameters=[
      Parameter(
        annotation=List[str],
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='argv',
      ),
    ],
    return_annotation=argparse.Namespace,
  )

  assert hasattr(lmp.script.eval_txt_ppl, 'main')
  assert inspect.isfunction(lmp.script.eval_txt_ppl.main)
  assert inspect.signature(lmp.script.eval_txt_ppl.main) == Signature(
    parameters=[
      Parameter(
        annotation=List[str],
        default=Parameter.empty,
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        name='argv',
      ),
    ],
    return_annotation=None,
  )
