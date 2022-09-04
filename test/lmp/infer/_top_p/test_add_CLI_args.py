"""Test parser arguments.

Test target:
- :py:meth:`lmp.infer._top_p.TopPInfer.add_CLI_args`.
"""

import argparse
import math

from lmp.infer._top_p import TopPInfer


def test_default_value() -> None:
  """Ensure default value consistency."""
  p = 0.9
  parser = argparse.ArgumentParser()
  TopPInfer.add_CLI_args(parser=parser)
  args = parser.parse_args([])

  assert math.isclose(args.p, p)


def test_arguments(p: int) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  TopPInfer.add_CLI_args(parser=parser)
  args = parser.parse_args([
    '--p',
    str(p),
  ])

  assert math.isclose(args.p, p)
