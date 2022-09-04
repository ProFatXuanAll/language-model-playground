"""Test parser arguments.

Test target:
- :py:meth:`lmp.infer._top_k.TopKInfer.add_CLI_args`.
"""

import argparse

from lmp.infer._top_k import TopKInfer


def test_default_value() -> None:
  """Ensure default value consistency."""
  k = 5
  parser = argparse.ArgumentParser()
  TopKInfer.add_CLI_args(parser=parser)
  args = parser.parse_args([])

  assert args.k == k


def test_arguments(k: int) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  TopKInfer.add_CLI_args(parser=parser)
  args = parser.parse_args([
    '--k',
    str(k),
  ])

  assert args.k == k
