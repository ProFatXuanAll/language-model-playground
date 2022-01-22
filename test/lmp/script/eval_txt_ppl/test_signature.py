"""Test :py:mod:`lmp.script.eval_txt_ppl` signatures."""

import argparse
import inspect
from inspect import Parameter, Signature
from typing import List

import lmp.script.eval_txt_ppl


def test_module_function() -> None:
  """Ensure module function's signatures."""
  assert inspect.isfunction(lmp.script.eval_txt_ppl.parse_args)
  assert inspect.signature(lmp.script.eval_txt_ppl.parse_args) == Signature(
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
  assert inspect.isfunction(lmp.script.eval_txt_ppl.main)
  assert inspect.signature(lmp.script.eval_txt_ppl.main) == Signature(
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
