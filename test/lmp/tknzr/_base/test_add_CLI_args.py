"""Test adding CLI arguments to parser.

Test target:
- :py:meth:`lmp.tknzr._base.BaseTknzr.add_CLI_args`.
"""

import argparse

from lmp.tknzr._base import BaseTknzr


def test_arguments() -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  BaseTknzr.add_CLI_args(parser=parser)
  assert parser.parse_args([]) == argparse.Namespace()
